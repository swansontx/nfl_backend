"""Simple caching layer for external API responses.

Uses in-memory caching with TTL (time-to-live) for different data types.
Can be upgraded to Redis in production.
"""

from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, Callable, Any
import json


class SimpleCache:
    """In-memory cache with TTL support."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if datetime.now() > entry['expires']:
            del self._cache[key]
            return None

        return entry['value']

    def set(self, key: str, value: Any, ttl_seconds: int):
        """Set value in cache with TTL."""
        self._cache[key] = {
            'value': value,
            'expires': datetime.now() + timedelta(seconds=ttl_seconds)
        }

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()

    def clear_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry['expires']
        ]
        for key in expired_keys:
            del self._cache[key]


# Global cache instance
cache = SimpleCache()


def cached(ttl_seconds: int):
    """Decorator to cache function results.

    Args:
        ttl_seconds: Time to live in seconds

    Example:
        @cached(ttl_seconds=900)  # 15 minutes
        def get_injuries():
            return expensive_api_call()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds)
            return result

        return wrapper
    return decorator


# Cache TTL constants (in seconds)
CACHE_TTL = {
    'injuries': 900,        # 15 minutes - injuries don't change often
    'weather': 3600,        # 1 hour - weather changes slowly
    'news': 300,            # 5 minutes - more dynamic
    'player_stats': 3600,   # 1 hour - stats are historical
    'sportsbook_lines': 60, # 1 minute - lines move frequently
    'projections': 1800,    # 30 minutes - model outputs
}
