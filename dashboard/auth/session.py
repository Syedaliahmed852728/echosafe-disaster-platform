"""
Session Management for Streamlit Dashboard.

Secure refresh-safe behavior:
- JWT is never stored in URL.
- Browser only gets an opaque session id cookie.
- Actual JWT remains server-side in Streamlit cache/session memory.
- Logout invalidates the server-side session and deletes the browser cookie.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from backend.config.security import JWTManager
from backend.config.logger import get_logger
from backend.config.settings import SETTINGS

logger = get_logger(__name__)

SESSION_KEY = "echosafe_session"
TOKEN_KEY = "echosafe_token"
USER_KEY = "echosafe_user"
AUTH_STATUS_KEY = "auth_status"
SESSION_ID_KEY = "echosafe_session_id"

AUTH_COOKIE_NAME = "echosafe_sid"
COOKIE_CLEAR_FLAG = "_echosafe_clear_auth_cookie"

# Match cookie lifetime to your JWT refresh period setting.
COOKIE_MAX_AGE_SECONDS = int(SETTINGS.security.refresh_token_days * 24 * 60 * 60)


@st.cache_resource
def _server_session_store() -> dict[str, dict[str, Any]]:
    """Server-side in-memory session registry.

    This survives browser refresh while the Streamlit server process is alive.
    For real production deployment, replace this with Redis/Postgres.
    """
    return {}


def _now() -> float:
    return time.time()


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions from the server-side store."""
    store = _server_session_store()
    max_age = COOKIE_MAX_AGE_SECONDS
    current_time = _now()

    expired = [
        sid
        for sid, record in store.items()
        if current_time - float(record.get("created_at", 0)) > max_age
    ]

    for sid in expired:
        store.pop(sid, None)


def _read_browser_cookie(name: str) -> str | None:
    """Read browser cookie through Streamlit context."""
    try:
        value = st.context.cookies.get(name)
        if value:
            return str(value)
    except Exception:
        return None

    return None


def _set_browser_cookie(name: str, value: str, max_age_seconds: int) -> None:
    """Set cookie using a small browser-side script."""
    js_name = json.dumps(name)
    js_value = json.dumps(value)
    js_max_age = int(max_age_seconds)

    components.html(
        f"""
        <script>
        document.cookie =
            {js_name} + "=" + encodeURIComponent({js_value}) +
            "; Max-Age=" + {js_max_age} +
            "; Path=/" +
            "; SameSite=Lax";
        </script>
        """,
        height=0,
        width=0,
    )


def _delete_browser_cookie(name: str) -> None:
    """Delete browser cookie using a small browser-side script."""
    js_name = json.dumps(name)

    components.html(
        f"""
        <script>
        document.cookie =
            {js_name} + "=; Max-Age=0; Path=/; SameSite=Lax";
        document.cookie =
            {js_name} + "=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/; SameSite=Lax";
        </script>
        """,
        height=0,
        width=0,
    )


def init_session() -> None:
    """Initialize session state variables."""
    if AUTH_STATUS_KEY not in st.session_state:
        st.session_state[AUTH_STATUS_KEY] = False

    if USER_KEY not in st.session_state:
        st.session_state[USER_KEY] = None

    if TOKEN_KEY not in st.session_state:
        st.session_state[TOKEN_KEY] = None

    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = None

    _cleanup_expired_sessions()


def restore_session_from_cookie() -> bool:
    """Restore authenticated session after browser refresh.

    This does NOT trust a JWT in the URL. It only accepts an opaque session id
    stored in a browser cookie and verifies that the server still has the JWT.
    """
    if st.session_state.get(TOKEN_KEY):
        return is_authenticated()

    sid = _read_browser_cookie(AUTH_COOKIE_NAME)
    if not sid:
        return False

    store = _server_session_store()
    record = store.get(sid)

    if not record:
        st.session_state[COOKIE_CLEAR_FLAG] = True
        return False

    token = record.get("token")
    username = record.get("username")

    if not token:
        store.pop(sid, None)
        st.session_state[COOKIE_CLEAR_FLAG] = True
        return False

    payload = JWTManager.decode_token(token)
    if payload is None:
        store.pop(sid, None)
        st.session_state[COOKIE_CLEAR_FLAG] = True
        return False

    st.session_state[AUTH_STATUS_KEY] = True
    st.session_state[TOKEN_KEY] = token
    st.session_state[USER_KEY] = username or payload.get("username") or payload.get("sub")
    st.session_state[SESSION_ID_KEY] = sid

    store[sid]["last_seen"] = _now()

    return True


def persist_auth_cookie() -> None:
    """Persist current server session id to browser cookie."""
    sid = st.session_state.get(SESSION_ID_KEY)

    if sid:
        _set_browser_cookie(AUTH_COOKIE_NAME, sid, COOKIE_MAX_AGE_SECONDS)


def clear_auth_cookie_if_requested() -> None:
    """Clear browser auth cookie after logout or invalid session."""
    if st.session_state.pop(COOKIE_CLEAR_FLAG, False):
        _delete_browser_cookie(AUTH_COOKIE_NAME)


def is_authenticated() -> bool:
    """Check if user is authenticated with valid token."""
    token = st.session_state.get(TOKEN_KEY)

    if not token:
        return False

    payload = JWTManager.decode_token(token)

    if payload is None:
        logout()
        return False

    return True


def login_user(username: str, token: str) -> None:
    """Store user session after successful login."""
    payload = JWTManager.decode_token(token)

    if payload is None:
        logger.warning("Rejected login for user=%s because JWT is invalid", username)
        return

    # Invalidate old server-side session for this browser tab if present.
    old_sid = st.session_state.get(SESSION_ID_KEY)
    if old_sid:
        _server_session_store().pop(old_sid, None)

    sid = secrets.token_urlsafe(32)

    _server_session_store()[sid] = {
        "username": username,
        "token": token,
        "created_at": _now(),
        "last_seen": _now(),
    }

    st.session_state[AUTH_STATUS_KEY] = True
    st.session_state[TOKEN_KEY] = token
    st.session_state[USER_KEY] = username
    st.session_state[SESSION_ID_KEY] = sid

    logger.info("User logged in: %s", username)


def logout() -> None:
    """Clear user session and invalidate server-side session."""
    sid = st.session_state.get(SESSION_ID_KEY) or _read_browser_cookie(AUTH_COOKIE_NAME)

    if sid:
        _server_session_store().pop(sid, None)

    st.session_state[AUTH_STATUS_KEY] = False
    st.session_state[TOKEN_KEY] = None
    st.session_state[USER_KEY] = None
    st.session_state[SESSION_ID_KEY] = None
    st.session_state[COOKIE_CLEAR_FLAG] = True

    logger.info("User logged out")


def get_current_user() -> dict:
    """Get current user info from token."""
    token = st.session_state.get(TOKEN_KEY)

    if token:
        payload = JWTManager.decode_token(token) or {}

        if payload and st.session_state.get(USER_KEY):
            payload.setdefault("username", st.session_state.get(USER_KEY))

        return payload

    return {}