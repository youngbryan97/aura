"""core/audit/__init__.py — Immutable autonomous action audit trail.

Re-exports the core AuditLog and coordinates adversarial self-audits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.runtime.sqlite_support import connection_is_open, open_tracked
from core.security.structural_redaction import redact_structure

logger = logging.getLogger("Aura.Audit")

_DB_PATH = config.paths.data_dir / "audit.db"
_AUDIT_RETRY_ERRORS = (sqlite3.DatabaseError, OSError, TypeError, ValueError)
#: Ceiling on rows a single query may materialise.
_MAX_QUERY_ROWS = 10_000

#: Columns added after the original schema. Existing databases are migrated
#: rather than recreated — recreating an audit log to change its shape is
#: the same act as deleting it.
_MIGRATIONS = (
    ("seq", "INTEGER"),
    ("prev_hash", "TEXT"),
    ("entry_hash", "TEXT"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id           TEXT PRIMARY KEY,
    action_type  TEXT NOT NULL,     -- 'skill_call' | 'autonomous_goal' | 'self_repair' | 'file_write' | etc.
    description  TEXT NOT NULL,
    actor        TEXT NOT NULL,     -- 'user' | 'autonomous' | 'terminal_monitor' | 'hephaestus'
    skill_name   TEXT,
    params       TEXT,              -- JSON, REDACTED (enforced in record(), not merely promised)
    result_ok    INTEGER,           -- 1=success, 0=failure, NULL=unknown
    cid          TEXT,              -- correlation ID
    session_id   TEXT,
    created_at   REAL NOT NULL,
    -- CP126 5b07b11d: the table had no update/delete prevention, signature,
    -- hash chain, sequence commitment or verifier, so any process with
    -- database access could alter or remove rows undetected. A row now
    -- commits to its predecessor, so a deletion or edit anywhere in the
    -- history breaks the chain at that point and verify_chain() names it.
    seq          INTEGER,
    prev_hash    TEXT,
    entry_hash   TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(action_type);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor);
"""


#: How SQLite says a database is unusable. "malformed" and "corrupt" alone
#: missed the most common one — opening a file that is not a database at all
#: reports "file is not a database", so a truncated or overwritten audit log
#: raised straight past the quarantine path and out of the constructor.
_CORRUPTION_MARKERS = (
    "malformed",
    "corrupt",
    "not a database",
    "encrypted or is not a database",
)


def _is_corruption(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CORRUPTION_MARKERS)


#: The chain's anchor. A first entry commits to this, so an attacker cannot
#: truncate the log to a shorter valid chain and start over.
_CHAIN_GENESIS = "aura.audit.chain.genesis.v1"


def _entry_digest(
    prev_hash, seq, entry_id, action_type, description, actor,
    skill_name, params_json, result_ok, cid, session_id, created_at,
) -> str:
    """Commit to this row AND its predecessor."""
    payload = json.dumps(
        [
            prev_hash, seq, entry_id, action_type, description, actor,
            skill_name, params_json, result_ok, cid, session_id, created_at,
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    """
    Append-only audit log. Once written, records are never modified.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = str(db_path or _DB_PATH)
        self._con: sqlite3.Connection | None = None
        # CP126 48b8237f: check_same_thread=False permitted concurrent
        # callers while connection creation, execute, commit, heal, close
        # and query were all unsynchronised.
        self._lock = checked_lock("audit.instance", reentrant=True)
        self._unavailable = False
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        # `is None` was not enough. A cached handle can be closed underneath
        # this store — by shutdown, by a test teardown, by corruption recovery
        # — and the old check returned the dead connection to every later
        # caller, which then raised "Cannot operate on a closed database"
        # forever after with no path back.
        if self._con is not None and not connection_is_open(self._con):
            self._con = None
        if self._con is None:
            con = open_tracked(self._db_path, timeout=10, check_same_thread=False)
            con.row_factory = sqlite3.Row
            try:
                con.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                # Do not cache a connection to an unusable file: the next
                # caller would reuse it and fail the same way, and the
                # quarantine path would never see a fresh open.
                con.close()
                raise
            self._con = con
            self._con.execute("PRAGMA synchronous=NORMAL")
        return self._con

    def close(self):
        if self._con:
            self._con.close()
            self._con = None

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release the connection. Never suppresses the exception.

        A durable store with no obvious way to be released gets released by
        the garbage collector instead — at an unpredictable moment, inside
        unrelated work. `with AuditLog(...)` makes the correct lifetime the
        easy one to write.
        """
        self.close()

    def _heal_database(self):
        """Quarantine a corrupt audit database. NEVER delete one.

        CP126 2a4030ee. This used to rename the main file and unlink the
        WAL and SHM sidecars — and if the rename failed, delete everything
        — then create a fresh empty database. That destroys the evidence at
        precisely the moment tampering has to be investigated, and the WAL
        holds the most RECENT records, which is the part an investigator
        wants most.

        A corrupt audit log is itself a finding. Everything is moved aside
        together, nothing is removed, and if the quarantine cannot be
        completed this refuses to start a fresh log over the old one rather
        than trading the evidence for availability.
        """
        self.close()
        db_file = Path(self._db_path)
        if not db_file.exists():
            self._build_schema()
            return

        stamp = int(time.time())
        quarantine = db_file.with_suffix(f".db.corrupt.{stamp}")
        try:
            db_file.rename(quarantine)
            # The sidecars are moved WITH the database, not deleted: the WAL
            # is where the newest entries live.
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(db_file) + suffix)
                if sidecar.exists():
                    sidecar.rename(Path(str(quarantine) + suffix))
            record_degradation(
                "audit",
                sqlite3.DatabaseError("audit database corrupt"),
                severity="critical",
                action=(
                    f"quarantined the corrupt audit log at {quarantine.name} and "
                    "started a new one; the old evidence is intact and a corrupt "
                    "audit log is itself a finding"
                ),
            )
            logger.critical("Corrupted audit database quarantined at %s", quarantine)
        except OSError as ex:
            # Refuse rather than delete. An audit log that cannot preserve
            # its predecessor must not quietly replace it.
            record_degradation(
                "audit",
                ex,
                severity="critical",
                action=(
                    "could not quarantine the corrupt audit database; REFUSED to "
                    "delete or overwrite it, so auditing is unavailable until an "
                    "operator moves the file aside"
                ),
            )
            logger.critical(
                "Refusing to delete the corrupt audit database at %s: %s", db_file, ex
            )
            self._unavailable = True
            return
        self._build_schema()

    #: SQLite's file magic. A database always starts with it.
    _SQLITE_MAGIC = b"SQLite format 3\x00"

    def _quarantine_if_not_a_database(self) -> None:
        """Check the header BEFORE opening, so the WAL survives.

        Learned the hard way: sqlite3.connect() on a file that is not a
        database deletes the stale ``-wal`` during the failed open, so by
        the time an exception reaches the quarantine path the most recent
        entries are already gone. Reading sixteen bytes first costs nothing
        and keeps the evidence whole.
        """
        db_file = Path(self._db_path)
        try:
            if not db_file.exists() or db_file.stat().st_size == 0:
                return
            with open(db_file, "rb") as handle:
                header = handle.read(len(self._SQLITE_MAGIC))
        except OSError:
            return
        if header == self._SQLITE_MAGIC:
            return
        logger.critical(
            "Audit database at %s is not a SQLite file; quarantining before open.",
            db_file,
        )
        self._heal_database()

    def _init(self):
        self._quarantine_if_not_a_database()
        if self._unavailable:
            return
        self._build_schema()

    def _build_schema(self):
        try:
            con = self._connect()
            con.executescript(_SCHEMA)
            # Older databases predate the hash chain. Add the columns rather
            # than recreating the table: recreating an audit log to change
            # its shape is the same act as deleting it.
            existing = {row[1] for row in con.execute("PRAGMA table_info(audit_log)")}
            for column, column_type in _MIGRATIONS:
                if column not in existing:
                    con.execute(f"ALTER TABLE audit_log ADD COLUMN {column} {column_type}")
            con.commit()
        except sqlite3.DatabaseError as e:
            if _is_corruption(e):
                logger.error("Audit database unusable on init: %s. Quarantining...", e)
                self._heal_database()
            else:
                raise

    def record(
        self,
        action_type: str,
        description: str,
        actor: str = "autonomous",
        skill_name: str | None = None,
        params: dict[str, Any] | None = None,
        result_ok: bool | None = None,
        cid: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Append one entry. Returns its id, or "" if it was NOT persisted.

        CP126 c16add51: every handled insert, commit, healing and retry
        failure fell through to ``return entry_id``, so a caller holding an
        id could not tell a durable audit receipt from a lost event — and
        an audit log that silently loses events is worse than none, because
        its silence is read as "nothing happened".

        CP126 a468408a: the schema comment promised redacted JSON and the
        code serialised the caller's dictionary verbatim with
        ``default=str``, storing credentials, tokens and personal data in
        the clear. Redaction is now performed here rather than promised in
        a comment.
        """
        entry_id = uuid.uuid4().hex  # CP126 9fb7e637: full id, not 48 bits
        if getattr(self, "_unavailable", False):
            return ""

        redacted_params = None
        if params:
            try:
                safe, _report = redact_structure(dict(params))
                redacted_params = json.dumps(safe, default=str)
            except (TypeError, ValueError) as exc:
                record_degradation(
                    "audit",
                    exc,
                    action="stored an audit entry without its parameters after redaction failed",
                )
                redacted_params = None

        created_at = time.time()
        with self._lock:
            try:
                return self._insert_locked(
                    entry_id, action_type, description, actor, skill_name,
                    redacted_params, result_ok, cid, session_id, created_at,
                )
            except sqlite3.DatabaseError as e:
                if _is_corruption(e):
                    logger.error("Audit database unusable on write: %s. Quarantining...", e)
                    self._heal_database()
                    if getattr(self, "_unavailable", False):
                        return ""
                    try:
                        return self._insert_locked(
                            entry_id, action_type, description, actor, skill_name,
                            redacted_params, result_ok, cid, session_id, created_at,
                        )
                    except _AUDIT_RETRY_ERRORS as retry_err:
                        record_degradation(
                            "audit",
                            retry_err,
                            severity="critical",
                            action="LOST an audit entry after quarantining a corrupt log",
                        )
                        logger.error("Failed to record audit entry after healing: %s", retry_err)
                        return ""
                record_degradation(
                    "audit",
                    e,
                    severity="critical",
                    action="LOST an audit entry; the action happened and is unrecorded",
                )
                logger.error("Failed to record audit entry: %s", e)
                return ""
            except OSError as e:
                record_degradation(
                    "audit",
                    e,
                    severity="critical",
                    action="LOST an audit entry; the action happened and is unrecorded",
                )
                logger.error("Failed to record audit entry: %s", e)
                return ""

    def _insert_locked(
        self, entry_id, action_type, description, actor, skill_name,
        params_json, result_ok, cid, session_id, created_at,
    ) -> str:
        """Insert one chained row. Caller holds the lock."""
        con = self._connect()
        row = con.execute(
            "SELECT seq, entry_hash FROM audit_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_seq = int(row["seq"] or 0) if row and row["seq"] is not None else 0
        prev_hash = (row["entry_hash"] if row else None) or _CHAIN_GENESIS
        seq = prev_seq + 1
        result_value = 1 if result_ok is True else (0 if result_ok is False else None)
        entry_hash = _entry_digest(
            prev_hash, seq, entry_id, action_type, description, actor,
            skill_name, params_json, result_value, cid, session_id, created_at,
        )
        con.execute(
            "INSERT INTO audit_log (id, action_type, description, actor, skill_name, "
            "params, result_ok, cid, session_id, created_at, seq, prev_hash, entry_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entry_id, action_type, description, actor, skill_name, params_json,
                result_value, cid, session_id, created_at, seq, prev_hash, entry_hash,
            ),
        )
        con.commit()
        return entry_id

    def verify_chain(self) -> dict[str, Any]:
        """Recompute the chain and report the first place it breaks.

        A deleted row breaks the sequence; an edited row breaks its own
        digest; a re-inserted row breaks its successor's ``prev_hash``.
        Rows written before the chain existed are reported as unchained
        rather than as violations — they are genuinely unprotected, and
        saying so is the point.
        """
        with self._lock:
            try:
                con = self._connect()
                rows = list(con.execute("SELECT * FROM audit_log ORDER BY seq ASC, created_at ASC"))
            except (sqlite3.DatabaseError, OSError) as exc:
                record_degradation("audit", exc, action="could not read the audit chain")
                return {"verified": False, "error": f"{type(exc).__name__}: {exc}"}

        unchained = sum(1 for row in rows if row["entry_hash"] is None)
        chained = [row for row in rows if row["entry_hash"] is not None]
        expected_prev = _CHAIN_GENESIS
        expected_seq = None
        for row in chained:
            if expected_seq is not None and int(row["seq"] or 0) != expected_seq + 1:
                return {
                    "verified": False,
                    "broken_at": row["id"],
                    "reason": f"sequence_gap:{expected_seq}->{row['seq']}",
                    "unchained_legacy_rows": unchained,
                }
            digest = _entry_digest(
                row["prev_hash"], row["seq"], row["id"], row["action_type"],
                row["description"], row["actor"], row["skill_name"], row["params"],
                row["result_ok"], row["cid"], row["session_id"], row["created_at"],
            )
            if digest != row["entry_hash"]:
                return {
                    "verified": False,
                    "broken_at": row["id"],
                    "reason": "entry_modified_after_write",
                    "unchained_legacy_rows": unchained,
                }
            if row["prev_hash"] != expected_prev:
                return {
                    "verified": False,
                    "broken_at": row["id"],
                    "reason": "predecessor_missing_or_altered",
                    "unchained_legacy_rows": unchained,
                }
            expected_prev = row["entry_hash"]
            expected_seq = int(row["seq"] or 0)
        return {
            "verified": True,
            "entries": len(chained),
            "unchained_legacy_rows": unchained,
        }

    def get_recent(
        self,
        limit: int = 100,
        action_type: str | None = None,
        actor: str | None = None,
    ) -> list[dict[str, Any]]:
        # CP126 5194bf9d: `limit` went to SQLite unvalidated, and a NEGATIVE
        # LIMIT means NO limit — a caller passing -1 pulled the entire audit
        # history into memory.
        try:
            bounded_limit = int(limit)
        except (TypeError, ValueError):
            bounded_limit = 100
        bounded_limit = max(1, min(bounded_limit, _MAX_QUERY_ROWS))
        limit = bounded_limit
        query = "SELECT * FROM audit_log"
        args = []
        conditions = []
        if action_type:
            conditions.append("action_type = ?")
            args.append(action_type)
        if actor:
            conditions.append("actor = ?")
            args.append(actor)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        try:
            con = self._connect()
            rows = con.execute(query, args).fetchall()
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error("Audit database is malformed on get_recent: %s. Healing...", e)
                self._heal_database()
                con = self._connect()
                rows = con.execute(query, args).fetchall()
            else:
                raise
        return [dict(r) for r in rows]

    def get_autonomous_summary(self, since_hours: float = 24.0) -> dict[str, Any]:
        """Summary of autonomous actions in the last N hours."""
        cutoff = time.time() - (since_hours * 3600)
        try:
            return self._get_autonomous_summary_internal(cutoff, since_hours)
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error("Audit database is malformed on summary: %s. Healing...", e)
                self._heal_database()
                return self._get_autonomous_summary_internal(cutoff, since_hours)
            raise

    def _get_autonomous_summary_internal(self, cutoff: float, since_hours: float) -> dict[str, Any]:
        con = self._connect()
        total = con.execute(
            "SELECT COUNT(*) FROM audit_log WHERE actor != 'user' AND created_at > ?",
            (cutoff,),
        ).fetchone()[0]
        by_type = con.execute(
            "SELECT action_type, COUNT(*) as c FROM audit_log "
            "WHERE actor != 'user' AND created_at > ? GROUP BY action_type",
            (cutoff,),
        ).fetchall()
        failures = con.execute(
            "SELECT COUNT(*) FROM audit_log "
            "WHERE actor != 'user' AND result_ok = 0 AND created_at > ?",
            (cutoff,),
        ).fetchone()[0]
        return {
            "period_hours": since_hours,
            "total_autonomous_actions": total,
            "failures": failures,
            "by_type": {r["action_type"]: r["c"] for r in by_type},
        }

    def get_skill_performance_stats(self, since_hours: float = 24.0) -> list[dict[str, Any]]:
        """Calculates performance statistics for each skill in the last N hours."""
        cutoff = time.time() - (since_hours * 3600)
        try:
            return self._get_skill_performance_stats_internal(cutoff)
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error("Audit database is malformed on stats: %s. Healing...", e)
                self._heal_database()
                return self._get_skill_performance_stats_internal(cutoff)
            raise

    def _get_skill_performance_stats_internal(self, cutoff: float) -> list[dict[str, Any]]:
        query = """
            SELECT 
                skill_name,
                COUNT(*) as calls,
                SUM(result_ok) as successes,
                AVG(CASE WHEN result_ok IS NOT NULL THEN 1.0 ELSE 0.0 END) as reporting_rate
            FROM audit_log
            WHERE action_type = 'skill_call' AND created_at > ? AND skill_name IS NOT NULL
            GROUP BY skill_name
        """
        con = self._connect()
        rows = con.execute(query, (cutoff,)).fetchall()
        
        stats = []
        for r in rows:
            calls = r["calls"]
            successes = r["successes"] or 0
            success_rate = (successes / calls) if calls > 0 else 1.0
            stats.append({
                "skill_name": r["skill_name"],
                "calls": calls,
                "successes": successes,
                "success_rate": success_rate,
                "reporting_rate": r["reporting_rate"]
            })
        return sorted(stats, key=lambda x: x["success_rate"])


_audit: AuditLog | None = None
_audit_lock = checked_lock("audit.module")


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        with _audit_lock:
            if _audit is None:
                _audit = AuditLog()
    return _audit


# Re-exports from adversarial modules
from core.audit.action_challenger import ActionChallenger
from core.audit.adversarial_auditor import AdversarialAuditor, get_adversarial_auditor
from core.audit.claim_challenger import ClaimChallenger
from core.audit.failure_injector import FailureInjector
from core.audit.red_team_agent import RedTeamAgent
