from core.runtime.errors import record_degradation
import logging
import sqlite3
import requests
import json
import os
from datetime import datetime
from typing import Optional

# Configuration (Ideally these would come from core.config)
DB_FILE = "data/aura_memory.db"
WEBHOOK_URL = os.environ.get("AURA_ALERTS_WEBHOOK")

class SQLiteMemoryHandler(logging.Handler):
    """
    Saves logs of INFO level and above to a persistent SQLite database.
    """
    def __init__(self, db_path: str = DB_FILE):
        super().__init__()
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    module TEXT,
                    message TEXT
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def emit(self, record):
        try:
            # We don't use the formatter for DB storage to keep raw data,
            # but we can if the formatting is complex.
            timestamp = datetime.now().isoformat()
            level = record.levelname
            module = record.module
            message = record.getMessage()

            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_logs (timestamp, level, module, message)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, level, module, message))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            record_degradation('aura_logging', e)
            # Prevent logging loops if the DB handler fails
            import sys
            print(f"FAILED TO WRITE TO MEMORY DB: {e}", file=sys.stderr)

class WebhookAlertHandler(logging.Handler):
    """
    Sends logs of ERROR level and above to a Discord/Slack webhook.
    """
    def __init__(self, webhook_url: Optional[str] = WEBHOOK_URL):
        super().__init__()
        self.webhook_url = webhook_url

    def emit(self, record):
        if not self.webhook_url:
            return

        try:
            log_entry = self.format(record)
            payload = {
                "content": f"🚨 **AURA CRITICAL ALERT** 🚨\n```text\n{log_entry}\n```"
            }
            # Short timeout to avoid hanging the main loop
            requests.post(self.webhook_url, json=payload, timeout=2.0)
        except Exception as e:
            record_degradation('aura_logging', e)
            import sys
            print(f"FAILED TO SEND WEBHOOK ALERT: {e}", file=sys.stderr)

def setup_enhanced_logging(logger_name: str = "Aura"):
    """
    Configures the given logger with standard console output, 
    SQLite persistent memory (INFO+), and Webhook alerts (ERROR+).
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Console Handler (DEBUG+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. SQLite Handler (INFO+)
    db_handler = SQLiteMemoryHandler()
    db_handler.setLevel(logging.INFO)
    # No formatter needed for DB as we store fields
    logger.addHandler(db_handler)

    # 3. Webhook Handler (ERROR+)
    if WEBHOOK_URL:
        alert_handler = WebhookAlertHandler(WEBHOOK_URL)
        alert_handler.setLevel(logging.ERROR)
        alert_handler.setFormatter(formatter)
        logger.addHandler(alert_handler)
        logger.info("📡 Webhook Alerting system active.")
    else:
        logger.info("Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).")

    return logger

# Singleton setup for the core logger
core_logger = setup_enhanced_logging("Aura.Core")
