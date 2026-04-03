import logging
from datetime import datetime, timezone
from typing import Optional, List, Any

from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from contextlib import contextmanager
from core.config import config

logger = logging.getLogger("Aura.ORM")
Base = declarative_base()

class SkillExecutionLog(Base):
    """Logs for every skill execution, ensuring no audit trail is lost."""
    __tablename__ = "skill_execution_logs"
    
    id = Column(Integer, primary_key=True)
    skill_name = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    params = Column(JSON)
    status = Column(String, index=True) # SUCCESS, FAILURE, TIMEOUT
    duration_ms = Column(Float)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)

class PersistentState:
    """
    Enterprise-grade state manager using SQLAlchemy.
    Provides a durable backbone for Aura's long-term memory and audit logs.
    """
    def __init__(self, db_url: Optional[str] = None):
        if not db_url:
            db_path = config.paths.data_dir / "zenith_state.db"
            db_url = f"sqlite:///{db_path}"
            
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        
        # Redact db_url for logging
        from urllib.parse import urlparse, urlunparse
        try:
            parsed = urlparse(db_url)
            if parsed.password:
                netloc = parsed.netloc.replace(parsed.password, "********")
                if parsed.username:
                    netloc = netloc.replace(parsed.username, "********")
                safe_db_url = urlunparse(parsed._replace(netloc=netloc))
            else:
                safe_db_url = db_url
        except Exception:
            safe_db_url = "[REDACTED DB URL]"
            
        logger.info("Durable ORM substrate initialized", db=safe_db_url)

    @contextmanager
    def _session_scope(self):
        """Private context manager for sessions with guaranteed cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_execution(
        self,
        skill_name: str,
        params: dict,
        status: str,
        duration_ms: float,
        result: Any = None,
        error: str = None,
    ) -> None:
        """Thread-safe logging of skill outcomes."""
        with self._session_scope() as session:
            log = SkillExecutionLog(
                skill_name=skill_name,
                params=params,
                status=status,
                duration_ms=duration_ms,
                result=result,
                error=error,
            )
            session.add(log)
            # commit is handled by _session_scope
