import os

try:
    from celery import Celery
except ImportError:
    Celery = None
from core.config import config

if Celery is None or not getattr(config.redis, "enabled", True):
    class MockCeleryConfig(dict):
        """Small Celery config surface used when Redis/Celery are unavailable."""

        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)
            return self

    class MockCelery:
        """Synchronous task shim preserving the Celery methods Aura uses."""

        def __init__(self):
            self.conf = MockCeleryConfig()
            self.sent_tasks = []

        def task(self, *args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]
            return lambda fn: fn

        def send_task(self, name, args=None, kwargs=None, **options):
            task = {
                "name": name,
                "args": tuple(args or ()),
                "kwargs": dict(kwargs or {}),
                "options": dict(options or {}),
            }
            self.sent_tasks.append(task)
            return task

    celery_app = MockCelery()
else:
    redis_url = os.environ.get("REDIS_URL", config.redis.url)
    celery_app = Celery("aura_zenith", broker=redis_url, backend=redis_url)

# Configuration for Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
)
