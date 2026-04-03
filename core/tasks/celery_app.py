import os
try:
    from celery import Celery
except ImportError:
    Celery = None
from core.config import config

if Celery is None or not getattr(config.redis, "enabled", True):
    class MockCelery:
        def task(self, *args, **kwargs):
            return lambda f: f
        def send_task(self, name, args=None, kwargs=None, **options):
            return None
        def conf(self): pass
        def update(self, *args, **kwargs): pass
    celery_app = MockCelery()
    celery_app.conf = MockCelery()
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
