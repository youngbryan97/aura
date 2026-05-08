import logging

logger = logging.getLogger(__name__)

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    logger.warning("⚠️ 'tenacity' not found. Using fallback NO-OP decorators.")
    
    def retry(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
        
    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

    def retry_if_exception_type(*args, **kwargs):
        return None
