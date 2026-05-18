import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import platform
import secrets
import time
import uuid as uuidlib
from collections.abc import Callable
from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.zenith_secrets import get_secret, set_secret

logger = logging.getLogger("Aura.Horcrux")
try:
    from cryptography.exceptions import InvalidTag
except ImportError:  # pragma: no cover - cryptography is a declared runtime dependency
    InvalidTag = ValueError

_HORCRUX_RECOVERABLE_ERRORS = (
    binascii.Error,
    FileNotFoundError,
    ImportError,
    InvalidTag,
    IsADirectoryError,
    json.JSONDecodeError,
    KeyError,
    OSError,
    PermissionError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_KEYCHAIN_ACCOUNT = "Aura"
SecretGetter = Callable[[str], str | None]
SecretSetter = Callable[[str, str], None]

# ═══════════════════════════════════════════════════════════
# GF(256) MATH FOR SHAMIR'S SECRET SHARING
# ═══════════════════════════════════════════════════════════
PRIME_POLY = 0x11B
EXP_TABLE = [0] * 512
LOG_TABLE = [0] * 256

x = 1
for i in range(255):
    EXP_TABLE[i] = x
    LOG_TABLE[x] = i
    
    x2 = x << 1
    if x2 & 0x100:
        x2 ^= PRIME_POLY
    x = x ^ x2

for i in range(255, 512):
    EXP_TABLE[i] = EXP_TABLE[i - 255]

def gf_add(x, y):
    return x ^ y

def gf_sub(x, y):
    return x ^ y

def gf_mul(x: int, y: int) -> int:
    if x == 0 or y == 0:
        return 0
    return int(EXP_TABLE[LOG_TABLE[x] + LOG_TABLE[y]])

def gf_div(x: int, y: int) -> int:
    if y == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if x == 0:
        return 0
    return int(EXP_TABLE[(LOG_TABLE[x] + 255 - LOG_TABLE[y]) % 255])

def eval_poly(poly, x):
    result = 0
    for coeff in reversed(poly):
        result = gf_add(gf_mul(result, x), coeff)
    return result

def split_secret(secret_bytes: bytes, threshold: int, num_shares: int) -> dict[int, bytes]:
    if threshold > num_shares:
        raise ValueError("Threshold > num_shares")
    shares: dict[int, bytearray] = {i: bytearray() for i in range(1, num_shares + 1)}
    for byte in secret_bytes:
        poly = [int(byte)] + [secrets.randbelow(256) for _ in range(threshold - 1)]
        for x in range(1, num_shares + 1):
            shares[x].append(eval_poly(poly, x))
    return {k: bytes(v) for k, v in shares.items()}

def lagrange_interpolate(x, x_s, y_s):
    num_points = len(x_s)
    result = 0
    for i in range(num_points):
        x_i = x_s[i]
        y_i = y_s[i]
        num = 1
        den = 1
        for j in range(num_points):
            if i == j:
                continue
            x_j = x_s[j]
            num = gf_mul(num, gf_add(x, x_j))
            den = gf_mul(den, gf_add(x_i, x_j))
        result = gf_add(result, gf_mul(y_i, gf_div(num, den)))
    return result

def reconstruct_secret(shares_dict: dict[int, bytes]) -> bytes:
    if not shares_dict:
        return b""
    x_s = list(shares_dict.keys())
    y_arrays = list(shares_dict.values())
    
    # Validation: all shards must have the same length
    first_shard = y_arrays[0]
    secret_len = len(first_shard)
    for i, y in enumerate(y_arrays):
        if len(y) != secret_len:
            raise ValueError(f"Inconsistent shard lengths: first shard is {secret_len} bytes, but shard {x_s[i]} is {len(y)} bytes.")
            
    secret = bytearray()
    for byte_idx in range(secret_len):
        y_s = [int(shard[byte_idx]) for shard in y_arrays]
        secret.append(lagrange_interpolate(0, x_s, y_s))
    return bytes(secret)

# ═══════════════════════════════════════════════════════════
# HARDWARE ENTANGLEMENT
# ═══════════════════════════════════════════════════════════
def _load_or_create_local_hardware_seed() -> bytes:
    """Return a stable private seed when hardware identifiers are unavailable."""
    seed_path = Path.home() / ".aura" / ".hardware_seed"
    try:
        if seed_path.exists():
            seed = seed_path.read_bytes().strip()
            if len(seed) >= 32:
                return hashlib.sha256(seed).digest()
            logger.warning("Horcrux local hardware seed was malformed; rotating it.")
    except _HORCRUX_RECOVERABLE_ERRORS as exc:
        record_degradation("horcrux", exc)
        logger.debug("Horcrux local hardware seed read failed: %s", exc)

    seed = secrets.token_bytes(32)
    try:
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(seed_path, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(seed)
    except _HORCRUX_RECOVERABLE_ERRORS as exc:
        record_degradation("horcrux", exc)
        logger.error("Horcrux could not persist a local hardware seed: %s", exc)
        return hashlib.sha256(b"fallback-aura-seed").digest()
    return hashlib.sha256(seed).digest()


def get_hardware_seed():
    """Derive a deterministic seed from native host signals without shelling out."""
    signals = []
    machine_id = _read_machine_id()
    if machine_id:
        signals.append(f"machine_id:{machine_id}")
    node = platform.node()
    if node:
        signals.append(f"node:{node}")
    uname = platform.uname()
    if uname.system or uname.machine:
        signals.append(f"platform:{uname.system}:{uname.machine}:{uname.processor}")
    mac_int = uuidlib.getnode()
    if mac_int:
        signals.append(f"uuid_getnode:{mac_int:012x}")

    if signals:
        return hashlib.sha256("|".join(signals).encode()).digest()
    return _load_or_create_local_hardware_seed()


def _keychain_secret_name(service: str) -> str:
    safe_service = "".join(ch if ch.isalnum() else "_" for ch in service.upper())
    return f"AURA_HORCRUX_{_KEYCHAIN_ACCOUNT}_{safe_service}"


def _read_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            if candidate.exists():
                value = candidate.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.debug("Horcrux machine-id read failed for %s: %s", candidate, exc)
    return ""

# ═══════════════════════════════════════════════════════════
# SHARD MANAGERS
# ═══════════════════════════════════════════════════════════
class HorcruxManager:
    threshold: int
    num_shares: int
    aura_dir: str
    hardware_base: bytes
    derived_key: bytes | None

    def __init__(
        self,
        base_dir: str | None = None,
        *,
        secret_getter: SecretGetter | None = None,
        secret_setter: SecretSetter | None = None,
    ):
        self.threshold = 3
        self.num_shares = 5
        self.aura_dir = os.path.expanduser(base_dir) if base_dir else os.path.expanduser("~/.aura")
        os.makedirs(self.aura_dir, exist_ok=True)
        self.hardware_base = get_hardware_seed()
        self.derived_key = None
        self._shard_cache_path = os.path.join(self.aura_dir, "shard_cache.enc")
        self._secret_getter = secret_getter or get_secret
        self._secret_setter = secret_setter or set_secret

    def _pack(self, x_index: int, shard_bytes: bytes) -> bytes:
        return bytes([x_index]) + shard_bytes
        
    def _unpack(self, packed: bytes) -> tuple[int | None, bytes | None]:
        if not packed:
            return None, None
        return int(packed[0]), packed[1:]

    # --- Shard 1: Keychain (Async) ---
    def _save_keychain_sync(self, x, shard, service="AuraHorcrux"):
        b64 = base64.b64encode(self._pack(x, shard)).decode()
        self._secret_setter(_keychain_secret_name(service), b64)

    async def _save_keychain(self, x, shard, service="AuraHorcrux"):
        await asyncio.to_thread(self._save_keychain_sync, x, shard, service)

    def _load_keychain_sync(self, service="AuraHorcrux"):
        try:
            value = self._secret_getter(_keychain_secret_name(service))
            if not value:
                return None, None
            return self._unpack(base64.b64decode(value.strip()))
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.debug("Keychain shard load failed for %s: %s", service, exc)
            return None, None

    async def _load_keychain(self, service="AuraHorcrux"):
        return await asyncio.to_thread(self._load_keychain_sync, service)

    # --- Shard Cache (Zenith Resilience) ---
    def _save_shard_cache(self, shards: dict[int, bytes]):
        """Save an encrypted shard cache for Keychain/offline recovery."""
        try:
            data = {str(k): base64.b64encode(v).decode() for k, v in shards.items()}
            raw = json.dumps(data).encode()
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            key = hashlib.sha256(self.hardware_base + b"aura-horcrux-shard-cache-v2").digest()
            nonce = secrets.token_bytes(12)
            ciphertext = AESGCM(key).encrypt(nonce, raw, b"horcrux-shard-cache")
            envelope = {
                "version": 2,
                "nonce": base64.b64encode(nonce).decode(),
                "ciphertext": base64.b64encode(ciphertext).decode(),
            }
            atomic_write_text(Path(self._shard_cache_path), json.dumps(envelope, sort_keys=True))
            os.chmod(self._shard_cache_path, 0o600)
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.error("Failed to save shard cache: %s", exc)

    def _load_shard_cache(self) -> dict[int, bytes]:
        """Loads shards from local fallback cache if Keychain is locked/offline."""
        try:
            cache_path = Path(self._shard_cache_path)
            if not cache_path.exists():
                return {}
            cache_bytes = cache_path.read_bytes()
            raw = self._decrypt_shard_cache(cache_bytes)
            data = json.loads(raw.decode())
            return {int(k): base64.b64decode(v) for k, v in data.items()}
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.debug("Shard cache load failed during startup: %s", exc)
            return {}

    def _decrypt_shard_cache(self, cache_bytes: bytes) -> bytes:
        try:
            envelope = json.loads(cache_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._decode_legacy_shard_cache(cache_bytes)
        if not isinstance(envelope, dict) or envelope.get("version") != 2:
            return self._decode_legacy_shard_cache(cache_bytes)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = hashlib.sha256(self.hardware_base + b"aura-horcrux-shard-cache-v2").digest()
        nonce = base64.b64decode(str(envelope["nonce"]))
        ciphertext = base64.b64decode(str(envelope["ciphertext"]))
        return AESGCM(key).decrypt(nonce, ciphertext, b"horcrux-shard-cache")

    def _decode_legacy_shard_cache(self, cache_bytes: bytes) -> bytes:
        mask = (self.hardware_base * (len(cache_bytes) // 32 + 1))[: len(cache_bytes)]
        return bytes(a ^ b for a, b in zip(cache_bytes, mask, strict=True))

    # --- Shard 2: Legacy Migration (Async) ---
    async def _save_zshrc(self, x, shard):
        logger.info("Migrating Shard 2 to secure Keychain storage.")
        await self._save_keychain(x, shard, service="AuraHorcrux_2")

    async def _load_zshrc(self):
        x, s = await self._load_keychain(service="AuraHorcrux_2")
        if x:
            return x, s
        
        # Fallback to legacy .zshrc if necessary (XOR'd during transition)
        return await asyncio.to_thread(self._load_legacy_env)

    def _load_legacy_env(self):
        try:
            val = os.environ.get("AURA_HORCRUX")
            if val:
                return self._unpack(base64.b64decode(val))
            zshrc = os.path.expanduser("~/.zshrc")
            if os.path.exists(zshrc):
                with open(zshrc) as f:
                    for line in f:
                        if line.startswith("export AURA_HORCRUX="):
                            parts = line.split("=", 1)
                            if len(parts) > 1:
                                b64 = parts[1].strip().strip("'").strip('"')
                                return self._unpack(base64.b64decode(b64))
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.debug("Legacy shard load failed: %s", exc)
        return None, None

    # --- Shard 3: ~/.aura/.core_seed (Async) ---
    async def _save_file(self, x, shard):
        await asyncio.to_thread(self._save_file_sync, x, shard)

    def _save_file_sync(self, x, shard):
        b64 = base64.b64encode(self._pack(x, shard)).decode()
        path = Path(self.aura_dir) / ".core_seed"
        atomic_write_text(path, b64)
        os.chmod(path, 0o600)

    async def _load_file(self):
        return await asyncio.to_thread(self._load_file_sync)

    def _load_file_sync(self):
        try:
            return self._unpack(base64.b64decode((Path(self.aura_dir) / ".core_seed").read_text().strip()))
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.debug("File shard load failed: %s", exc)
            return None, None

    # --- Dead Man's Recovery (Async) ---
    async def _load_hint(self, response):
        return await asyncio.to_thread(self._load_hint_sync, response)

    def _load_hint_sync(self, response):
        try:
            enc = base64.b64decode((Path(self.aura_dir) / ".hint_seed").read_text().strip())
            h = hashlib.sha256(response.encode()).digest()
            mask = (h * (len(enc)//32 + 1))[:len(enc)]
            dec = bytes(a ^ b for a, b in zip(enc, mask, strict=True))
            return self._unpack(dec)
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.debug("Hint shard load failed: %s", exc)
            return None, None

    # --- Core Logic (ZENITH: ASYNC) ---
    async def initialize(self, recovery_phrase=None, hint_answer=None) -> bool:
        """Harvest shards using async non-blocking I/O. Fallback to cache if Keychain stalls."""
        tasks = [
            self._load_keychain(),
            self._load_zshrc(),
            self._load_file()
        ]
        results = await asyncio.gather(*tasks)
        
        primary_shards = {}
        for x, s in results:
            if x:
                primary_shards[x] = s
        raw_shards = dict(primary_shards)
        
        if hint_answer:
            x4, s4 = await self._load_hint(hint_answer)
            if x4:
                raw_shards[x4] = s4
            
        if recovery_phrase:
            x5, s5 = self._load_mnemonic(recovery_phrase)
            if x5:
                raw_shards[x5] = s5

        # --- CACHE FALLBACK ---
        if len(raw_shards) < self.threshold:
            logger.warning("Keychain/File harvest insufficient (%d/%d). Checking Shard Cache...", 
                           len(raw_shards), self.threshold)
            cached = self._load_shard_cache()
            for k, v in cached.items():
                if k not in raw_shards:
                    raw_shards[k] = v

        # Consistency verification
        shards: dict[int, bytes] = {}
        if raw_shards:
            lengths: dict[int, list[int]] = {}
            for sid, data in raw_shards.items():
                if data:
                    shard_len = len(data)
                    lengths.setdefault(shard_len, []).append(sid)
            if lengths:
                best_len = max(lengths.keys(), key=lambda k: len(lengths[k]))
                for sid in lengths[best_len]:
                    shards[sid] = raw_shards[sid]

        if len(shards) >= self.threshold:
            master_key = reconstruct_secret(shards)
            self.derived_key = hashlib.sha256(master_key + self.hardware_base).digest()
            logger.info("✓ Master Key reconstructed (%d shards).", len(shards))
            # Background healing and cache update
            await self.heal(master_key, primary_shards)
            self._save_shard_cache(shards)
            return True

        # DEV mode fallback
        from core.config import Environment, config
        if config.env == Environment.DEV and shards:
            logger.warning("⚠️ [DEV MODE] Recovering from %d shards.", len(shards))
            master_key = reconstruct_secret(shards)
            self.derived_key = hashlib.sha256(master_key + self.hardware_base).digest()
            await self.heal(master_key, primary_shards)
            return True

        if not shards:
            logger.warning("No shards found. Initializing new Horcrux.")
            mnemonic = await self.shatter_new(hint_answer)
            print(f"\n[SECURITY] RECOVERY PHRASE: {mnemonic}\n")
            return True

        logger.error("Critical: Insufficient shards (%d). System locked.", len(shards))
        return False

    async def shatter_new(self, hint_answer=None):
        if not hint_answer:
             hint_answer = f"aura-default-{int(time.time())}"
        master_key = secrets.token_bytes(32)
        self.derived_key = hashlib.sha256(master_key + self.hardware_base).digest()
        shares = split_secret(master_key, self.threshold, self.num_shares)
        
        await asyncio.gather(
            self._save_keychain(1, shares[1]),
            self._save_zshrc(2, shares[2]),
            self._save_file(3, shares[3])
        )
        self._save_shard_cache({1: shares[1], 2: shares[2], 3: shares[3]})
        
        # Sync parts remain sync.
        self._save_hint_sync(shares[4], hint_answer)
        
        return self._generate_mnemonic(5, shares[5])

    async def heal(self, master_key, current_shards):
        if not master_key:
            return
        shares = split_secret(master_key, self.threshold, self.num_shares)
        tasks = []
        if 1 not in current_shards:
            tasks.append(self._save_keychain(1, shares[1]))
        if 2 not in current_shards:
            tasks.append(self._save_zshrc(2, shares[2]))
        if 3 not in current_shards:
            tasks.append(self._save_file(3, shares[3]))
        if tasks:
            await asyncio.gather(*tasks)

    def _save_hint_sync(self, shard, response):
        """Helper for shatter_new."""
        h = hashlib.sha256(response.encode()).digest()
        packed = self._pack(4, shard) # Fixed index
        mask = (h * (len(packed)//32 + 1))[:len(packed)]
        enc = bytearray(a ^ b for a, b in zip(packed, mask, strict=True))
        path = Path(self.aura_dir) / ".hint_seed"
        atomic_write_text(path, base64.b64encode(enc).decode())
        os.chmod(path, 0o600)

    def _generate_mnemonic(self, x, shard):
        packed = self._pack(x, shard)
        return packed.hex()
        
    def _load_mnemonic(self, hex_str):
        try:
            packed = bytes.fromhex(hex_str.strip())
            return self._unpack(packed)
        except _HORCRUX_RECOVERABLE_ERRORS as exc:
            record_degradation("horcrux", exc)
            logger.debug("Mnemonic shard load failed: %s", exc)
            return None, None

    async def check_shards(self) -> dict[int, bool]:
        """Async status check for all shards."""
        status = {}
        x1, _ = await self._load_keychain()
        status[1] = x1 is not None
        x2, _ = await self._load_keychain(service="AuraHorcrux_2")
        status[2] = x2 is not None
        x3, _ = await self._load_file()
        status[3] = x3 is not None
        status[4] = await asyncio.to_thread(
            os.path.exists,
            os.path.join(self.aura_dir, ".hint_seed"),
        )
        status[5] = True
        return status

    def get_key_string(self) -> str:
        if not self.derived_key:
            raise RuntimeError("Horcrux not initialized")
        return base64.b64encode(self.derived_key).decode()
