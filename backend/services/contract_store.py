"""Small in-memory cache for recently accessed contracts."""

from typing import Any


CONTRACT_STORE: dict[str, dict[str, Any]] = {}