from core.runtime.errors import record_degradation
import os
import secrets
import subprocess
import hashlib
import base64
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger("Aura.Horcrux")

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

def split_secret(secret_bytes: bytes, threshold: int, num_shares: int) -> Dict[int, bytes]:
    if threshold > num_shares:
        raise ValueError("Threshold > num_shares")
    shares: Dict[int, bytearray] = {i: bytearray() for i in range(1, num_shares + 1)}
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
            if i == j: continue
            x_j = x_s[j]
            num = gf_mul(num, gf_add(x, x_j))
            den = gf_mul(den, gf_add(x_i, x_j))
        result = gf_add(result, gf_mul(y_i, gf_div(num, den)))
    return result

def reconstruct_secret(shares_dict: Dict[int, bytes]) -> bytes:
    if not shares_dict: return b""
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
def get_hardware_seed():
    """Derives a deterministic seed from Mac hardware."""
    try:
        uuid_out = subprocess.check_output(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True, stderr=subprocess.DEVNULL)
        uuid = ""
        for line in uuid_out.split('\n'):
            if "IOPlatformUUID" in line:
                uuid = line.split('=')[1].strip().strip('"')
                break
        try:
            mac_out = subprocess.check_output(["ifconfig", "en0"], text=True, stderr=subprocess.DEVNULL)
            mac = ""
            for line in mac_out.split('\n'):
                if "ether" in line:
                    mac = line.split()[1].strip()
                    break
        except Exception:
            mac = "00:00:00:00:00:00"
        return hashlib.sha256(f"{uuid}-{mac}".encode()).digest()
    except Exception:
        return hashlib.sha256(b"fallback-aura-seed").digest()

# ═══════════════════════════════════════════════════════════
# SHARD MANAGERS
# ═══════════════════════════════════════════════════════════
class HorcruxManager:
    threshold: int
    num_shares: int
    aura_dir: str
    hardware_base: bytes
    derived_key: Optional[bytes]

    def __init__(self, base_dir: Optional[str] = None):
        self.threshold = 3
        self.num_shares = 5
        self.aura_dir = os.path.expanduser(base_dir) if base_dir else os.path.expanduser("~/.aura")
        os.makedirs(self.aura_dir, exist_ok=True)
        self.hardware_base = get_hardware_seed()
        self.derived_key = None
        self._shard_cache_path = os.path.join(self.aura_dir, "shard_cache.enc")

    def _pack(self, x_index: int, shard_bytes: bytes) -> bytes:
        return bytes([x_index]) + shard_bytes
        
    def _unpack(self, packed: bytes) -> tuple[Optional[int], Optional[bytes]]:
        if not packed: return None, None
        return int(packed[0]), packed[1:]

    # --- Shard 1: Keychain (Async) ---
    def _save_keychain_sync(self, x, shard, service="AuraHorcrux"):
        b64 = base64.b64encode(self._pack(x, shard)).decode()
        subprocess.run(["security", "add-generic-password", "-a", "Aura", "-s", service, "-w", b64, "-U"], check=False, stderr=subprocess.DEVNULL)

    async def _save_keychain(self, x, shard, service="AuraHorcrux"):
        await asyncio.to_thread(self._save_keychain_sync, x, shard, service)

    def _load_keychain_sync(self, service="AuraHorcrux"):
        try:
            out = subprocess.check_output(["security", "find-generic-password", "-a", "Aura", "-s", service, "-w"], text=True, stderr=subprocess.DEVNULL).strip()
            return self._unpack(base64.b64decode(out))
        except Exception:
            return None, None

    async def _load_keychain(self, service="AuraHorcrux"):
        return await asyncio.to_thread(self._load_keychain_sync, service)

    # --- Shard Cache (Zenith Resilience) ---
    def _save_shard_cache(self, shards: Dict[int, bytes]):
        """Saves an obfuscated copy of shards to ~/.aura/shard_cache.enc."""
        try:
            import json
            data = {str(k): base64.b64encode(v).decode() for k, v in shards.items()}
            raw = json.dumps(data).encode()
            # Simple hardware-bound XOR obfuscation (upgrade to AES in Phase 7)
            mask = (self.hardware_base * (len(raw)//32 + 1))[:len(raw)]
            obfuscated = bytes(a ^ b for a, b in zip(raw, mask))
            with open(self._shard_cache_path, "wb") as f:
                f.write(obfuscated)
        except Exception as e:
            record_degradation('horcrux', e)
            logger.error("Failed to save shard cache: %s", e)

    def _load_shard_cache(self) -> Dict[int, bytes]:
        """Loads shards from local fallback cache if Keychain is locked/offline."""
        try:
            if not os.path.exists(self._shard_cache_path):
                return {}
            import json
            with open(self._shard_cache_path, "rb") as f:
                obfuscated = f.read()
            mask = (self.hardware_base * (len(obfuscated)//32 + 1))[:len(obfuscated)]
            raw = bytes(a ^ b for a, b in zip(obfuscated, mask))
            data = json.loads(raw.decode())
            return {int(k): base64.b64decode(v) for k, v in data.items()}
        except Exception as e:
            record_degradation('horcrux', e)
            logger.debug("Shard cache load failed (expected during first boot): %s", e)
            return {}

    # --- Shard 2: Legacy Migration (Async) ---
    async def _save_zshrc(self, x, shard):
        logger.info("Migrating Shard 2 to secure Keychain storage.")
        await self._save_keychain(x, shard, service="AuraHorcrux_2")

    async def _load_zshrc(self):
        x, s = await self._load_keychain(service="AuraHorcrux_2")
        if x: return x, s
        
        # Fallback to legacy .zshrc if necessary (XOR'd during transition)
        return await asyncio.to_thread(self._load_legacy_env)

    def _load_legacy_env(self):
        try:
            val = os.environ.get("AURA_HORCRUX")
            if val:
                return self._unpack(base64.b64decode(val))
            zshrc = os.path.expanduser("~/.zshrc")
            if os.path.exists(zshrc):
                with open(zshrc, "r") as f:
                    for line in f:
                        if line.startswith("export AURA_HORCRUX="):
                            parts = line.split("=", 1)
                            if len(parts) > 1:
                                b64 = parts[1].strip().strip("'").strip('"')
                                return self._unpack(base64.b64decode(b64))
        except Exception as _exc:
            record_degradation('horcrux', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return None, None

    # --- Shard 3: ~/.aura/.core_seed (Async) ---
    async def _save_file(self, x, shard):
        await asyncio.to_thread(self._save_file_sync, x, shard)

    def _save_file_sync(self, x, shard):
        b64 = base64.b64encode(self._pack(x, shard)).decode()
        with open(os.path.join(self.aura_dir, ".core_seed"), "w") as f:
            f.write(b64)

    async def _load_file(self):
        return await asyncio.to_thread(self._load_file_sync)

    def _load_file_sync(self):
        try:
            with open(os.path.join(self.aura_dir, ".core_seed"), "r") as f:
                return self._unpack(base64.b64decode(f.read().strip()))
        except Exception:
            return None, None

    # --- Dead Man's Recovery (Async) ---
    async def _load_hint(self, response):
        return await asyncio.to_thread(self._load_hint_sync, response)

    def _load_hint_sync(self, response):
        try:
            with open(os.path.join(self.aura_dir, ".hint_seed"), "r") as f:
                enc = base64.b64decode(f.read().strip())
            h = hashlib.sha256(response.encode()).digest()
            dec = bytes(a ^ b for a, b in zip(enc, (h * (len(enc)//32 + 1))[:len(enc)]))
            return self._unpack(dec)
        except Exception:
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
            if x4: raw_shards[x4] = s4
            
        if recovery_phrase:
            x5, s5 = self._load_mnemonic(recovery_phrase)
            if x5: raw_shards[x5] = s5

        # --- CACHE FALLBACK ---
        if len(raw_shards) < self.threshold:
            logger.warning("Keychain/File harvest insufficient (%d/%d). Checking Shard Cache...", 
                           len(raw_shards), self.threshold)
            cached = self._load_shard_cache()
            for k, v in cached.items():
                if k not in raw_shards:
                    raw_shards[k] = v

        # Consistency verification
        shards: Dict[int, bytes] = {}
        if raw_shards:
            lengths: Dict[int, List[int]] = {}
            for sid, data in raw_shards.items():
                if data:
                    l = len(data)
                    lengths.setdefault(l, []).append(sid)
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
        from core.config import config, Environment
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
        
        # Sync parts remain sync
        self._save_hint_sync(shares[4], hint_answer) # Corrected arg order? Wait, check legacy.
        # Original was: self._save_hint(4, shares[4], response=hint_answer)
        # I'll keep it consistent.
        
        return self._generate_mnemonic(5, shares[5])

    async def heal(self, master_key, current_shards):
        if not master_key: return
        shares = split_secret(master_key, self.threshold, self.num_shares)
        tasks = []
        if 1 not in current_shards: tasks.append(self._save_keychain(1, shares[1]))
        if 2 not in current_shards: tasks.append(self._save_zshrc(2, shares[2]))
        if 3 not in current_shards: tasks.append(self._save_file(3, shares[3]))
        if tasks:
            await asyncio.gather(*tasks)

    def _save_hint_sync(self, shard, response):
        """Helper for shatter_new."""
        h = hashlib.sha256(response.encode()).digest()
        packed = self._pack(4, shard) # Fixed index
        enc = bytearray(a ^ b for a, b in zip(packed, (h * (len(packed)//32 + 1))[:len(packed)]))
        with open(os.path.join(self.aura_dir, ".hint_seed"), "w") as f:
            f.write(base64.b64encode(enc).decode())

    def _generate_mnemonic(self, x, shard):
        packed = self._pack(x, shard)
        return packed.hex()
        
    def _load_mnemonic(self, hex_str):
        try:
            packed = bytes.fromhex(hex_str.strip())
            return self._unpack(packed)
        except Exception:
            return None, None

    async def check_shards(self) -> Dict[int, bool]:
        """Async status check for all shards."""
        status = {}
        x1, _ = await self._load_keychain()
        status[1] = x1 is not None
        x2, _ = await self._load_keychain(service="AuraHorcrux_2")
        status[2] = x2 is not None
        x3, _ = await self._load_file()
        status[3] = x3 is not None
        status[4] = os.path.exists(os.path.join(self.aura_dir, ".hint_seed"))
        status[5] = True
        return status

    def get_key_string(self) -> str:
        if not self.derived_key:
            raise RuntimeError("Horcrux not initialized")
        return base64.b64encode(self.derived_key).decode()
