import hashlib
import hmac
import logging
import os
import time
from core.config import config

logger = logging.getLogger("Aura.AuditLog")

LOG_KEY_PATH = str(config.paths.home_dir / "keys/log_hmac_key.bin")
LOG_PATH = str(config.paths.home_dir / "logs/aura_audit.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_KEY_PATH), exist_ok=True)

if not os.path.exists(LOG_KEY_PATH):
    with open(LOG_KEY_PATH, "wb") as f:
        f.write(os.urandom(32))

# Chain state: hash of the previous log line for tamper-evident chaining
_prev_hash = "0" * 64  # Genesis hash

def append_audit(entry: str):
    """Appends an entry to the audit log with an HMAC signature.
    Each entry links to the previous entry's hash for tamper-evident chaining.
    """
    global _prev_hash
    ts = time.time()
    line = f"{ts}|{_prev_hash}|{entry}"
    
    try:
        with open(LOG_KEY_PATH, "rb") as f:
            key = f.read()
        
        # Use hmac.HMAC instead of deprecated hmac.new (BUG-005)
        sig = hmac.HMAC(key, line.encode("utf-8"), hashlib.sha256).hexdigest()
        _prev_hash = sig  # Chain to next entry
        
        with open(LOG_PATH, "a") as f:
            f.write(f"{line} | HM:{sig}\n")
            
    except Exception as e:
        logger.error("AUDIT LOG FAILURE: %s", e)
        try:
            with open(LOG_PATH, "a") as f:
                f.write(f"{ts}|LOG_FAILURE|{e}\n")
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)