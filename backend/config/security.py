"""
EchoSafe authentication primitives.

Two-tier auth model:

* **Project users** — created in advance and persisted to
  ``data/auth/project_users.json`` (PBKDF2-hashed). The login page calls
  :func:`authenticate_user`; the file is seeded with a default ``admin`` /
  ``echosafe`` account on first import so the demo can be opened immediately.
* **Local sign-ups** — handled inside ``dashboard/auth/login.py`` against a
  separate ``data/auth/local_users.json`` file. That layer is independent of
  this module.

JWTs are issued via :class:`JWTManager`. The signing secret comes from
``SETTINGS.security.jwt_secret`` (env: ``JWT_SECRET``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets as _secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import jwt

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS

logger = get_logger(__name__)

PROJECT_USERS_FILE: Path = SETTINGS.project_root / "data" / "auth" / "project_users.json"
PBKDF2_ITERATIONS = SETTINGS.security.password_iterations


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
    """PBKDF2-SHA256 hash; returns hex salt + hex hash for JSON persistence."""
    if salt is None:
        salt = _secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return {"salt": salt.hex(), "hash": derived.hex()}


def _verify_password(password: str, stored_salt_hex: str, stored_hash_hex: str) -> bool:
    salt = bytes.fromhex(stored_salt_hex)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(derived.hex(), stored_hash_hex)


# ---------------------------------------------------------------------------
# Project user store
# ---------------------------------------------------------------------------

_DEFAULT_USERS: Dict[str, Dict[str, Any]] = {
    # password = "echosafe" — for the demo / university presentation only.
    "admin": {"user_id": "u_admin_1", "role": "admin", "plain_password": "echosafe"},
    "analyst": {"user_id": "u_analyst_1", "role": "analyst", "plain_password": "echosafe"},
}


def _seed_default_users() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for username, meta in _DEFAULT_USERS.items():
        hashed = _hash_password(meta["plain_password"])
        out[username] = {
            "user_id": meta["user_id"],
            "username": username,
            "role": meta["role"],
            "salt": hashed["salt"],
            "hash": hashed["hash"],
        }
    return out


def _load_project_users() -> Dict[str, Any]:
    PROJECT_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not PROJECT_USERS_FILE.exists():
        seeded = _seed_default_users()
        PROJECT_USERS_FILE.write_text(json.dumps(seeded, indent=2))
        logger.info("Seeded default project users at %s", PROJECT_USERS_FILE)
        return seeded
    try:
        return json.loads(PROJECT_USERS_FILE.read_text())
    except Exception as exc:  # corrupted file
        logger.warning("Project users file unreadable, reseeding: %s", exc)
        seeded = _seed_default_users()
        PROJECT_USERS_FILE.write_text(json.dumps(seeded, indent=2))
        return seeded


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validate credentials against the project user store.

    Returns ``{user_id, username, role}`` on success, ``None`` otherwise.
    """
    if not username or not password:
        return None
    users = _load_project_users()
    record = users.get(username.strip())
    if not record:
        return None
    if not _verify_password(password, record["salt"], record["hash"]):
        return None
    return {
        "user_id": record["user_id"],
        "username": record["username"],
        "role": record.get("role", "analyst"),
    }


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

class JWTManager:
    """Thin wrapper around PyJWT using ``SETTINGS.security``."""

    @staticmethod
    def _secret() -> str:
        return SETTINGS.security.jwt_secret

    @staticmethod
    def _algorithm() -> str:
        return SETTINGS.security.jwt_algorithm

    @classmethod
    def create_access_token(
        cls, user_id: str, username: str, role: str = "analyst"
    ) -> str:
        now = datetime.now(tz=timezone.utc)
        payload = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int(
                (
                    now
                    + timedelta(minutes=SETTINGS.security.access_token_minutes)
                ).timestamp()
            ),
        }
        return jwt.encode(payload, cls._secret(), algorithm=cls._algorithm())

    @classmethod
    def decode_token(cls, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        try:
            return jwt.decode(token, cls._secret(), algorithms=[cls._algorithm()])
        except jwt.ExpiredSignatureError:
            logger.debug("JWT expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.debug("JWT invalid: %s", exc)
            return None


__all__ = ["JWTManager", "authenticate_user"]
