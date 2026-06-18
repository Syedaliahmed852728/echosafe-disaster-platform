"""
Login page for EchoSafe Dashboard.

Only this file is updated:
- Clean professional green / white / off-white frontend.
- Login, Sign Up, Forgot Password, and Reset Password are shown one screen at a time.
- No broken tabs/radio controls.
- No blank extra card.
- Clear text, clear buttons, clean input bars.
- Existing project authentication remains supported.
- New users can sign up and login through locally stored PBKDF2 credentials.
- Forgot password sends email if SMTP is configured; otherwise provides a local dev reset token.

This file does not touch pipelines, ML models, prediction files, app.py, or dashboard pages.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.auth.session import login_user
from backend.config.logger import get_logger
from backend.config.settings import SETTINGS

logger = get_logger(__name__)


LOCAL_AUTH_DIR = SETTINGS.project_root / "data" / "auth"
LOCAL_USERS_FILE = LOCAL_AUTH_DIR / "local_users.json"
RESET_TOKENS_FILE = LOCAL_AUTH_DIR / "password_reset_tokens.json"

PBKDF2_ITERATIONS = 210_000
RESET_TOKEN_MINUTES = 30


LOGIN_CSS = """
<style>
/* ---------------------------------------------------------
   GLOBAL AUTH PAGE CLEANUP
--------------------------------------------------------- */
[data-testid="stSidebar"],
[data-testid="stSidebarNav"] {
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    visibility: hidden !important;
}

#MainMenu,
footer {
    visibility: hidden !important;
}

/* ---------------------------------------------------------
   PAGE BACKGROUND
--------------------------------------------------------- */
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(0, 102, 51, 0.12), transparent 30%),
        radial-gradient(circle at 92% 14%, rgba(20, 148, 71, 0.10), transparent 34%),
        linear-gradient(135deg, #ffffff 0%, #f8fbf7 46%, #eef8f1 100%) !important;
    color-scheme: light !important;
}

.block-container {
    max-width: 1120px !important;
    padding-top: 3.1rem !important;
    padding-bottom: 2.5rem !important;
}

/* ---------------------------------------------------------
   MAIN LAYOUT
--------------------------------------------------------- */
.auth-layout {
    display: grid;
    grid-template-columns: 1.05fr 0.82fr;
    gap: 2.25rem;
    align-items: center;
    min-height: 76vh;
}

/* ---------------------------------------------------------
   LEFT BRAND PANEL
--------------------------------------------------------- */
.brand-panel {
    position: relative;
    min-height: 590px;
    border-radius: 34px;
    padding: 3.15rem;
    overflow: hidden;
    background:
        radial-gradient(circle at 87% 12%, rgba(255,255,255,0.22), transparent 32%),
        linear-gradient(135deg, #064e2f 0%, #0f7a3a 55%, #139447 100%);
    border: 1px solid rgba(6, 78, 47, 0.28);
    box-shadow: 0 30px 90px rgba(6, 78, 47, 0.25);
}

.brand-panel::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.055) 1px, transparent 1px);
    background-size: 42px 42px;
    animation: gridMove 18s linear infinite;
    mask-image: radial-gradient(circle at 42% 32%, black, transparent 78%);
}

.brand-panel::after {
    content: "";
    position: absolute;
    width: 430px;
    height: 430px;
    right: -150px;
    top: -140px;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    box-shadow: -58px 56px 0 rgba(6, 78, 47, 0.50);
    opacity: 0.78;
    animation: floatShape 7s ease-in-out infinite;
}

@keyframes gridMove {
    from { background-position: 0 0, 0 0; }
    to { background-position: 42px 42px, 42px 42px; }
}

@keyframes floatShape {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(12px); }
}

.brand-content {
    position: relative;
    z-index: 2;
}

.brand-pill {
    display: inline-flex;
    align-items: center;
    gap: .55rem;
    padding: .55rem .9rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    color: #ffffff;
    font-size: .76rem;
    font-weight: 950;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.brand-dot {
    width: .62rem;
    height: .62rem;
    border-radius: 999px;
    background: #ffffff;
    animation: pulseDot 1.8s infinite;
}

@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(255,255,255,.46); }
    70% { box-shadow: 0 0 0 12px rgba(255,255,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
}

.brand-title {
    margin: 2.35rem 0 1rem 0;
    color: #ffffff;
    font-size: clamp(3.2rem, 5vw, 5.2rem);
    line-height: .94;
    font-weight: 950;
    letter-spacing: -.075em;
}

.brand-subtitle {
    max-width: 650px;
    margin: 0;
    color: rgba(255,255,255,0.86);
    font-size: 1.06rem;
    line-height: 1.8;
}

.brand-footer {
    position: absolute;
    z-index: 2;
    left: 3.15rem;
    right: 3.15rem;
    bottom: 2.9rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(255,255,255,0.20);
    color: rgba(255,255,255,0.78);
    font-size: .92rem;
    line-height: 1.65;
}

/* ---------------------------------------------------------
   RIGHT AUTH AREA
--------------------------------------------------------- */
.auth-header {
    margin: 0 0 1.2rem 0 !important;
    padding-left: .1rem !important;
}

/* Shield/safeguard icon fully removed */
.auth-logo {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.auth-title {
    color: #102118 !important;
    margin: 0 !important;
    font-size: 2.32rem !important;
    line-height: 1.05 !important;
    font-weight: 950 !important;
    letter-spacing: -0.06em !important;
}

.auth-subtitle {
    color: #526158 !important;
    margin: .72rem 0 0 0 !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    font-weight: 550 !important;
}

/* ---------------------------------------------------------
   FORM CARD
--------------------------------------------------------- */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.98) !important;
    border: 1px solid rgba(15, 122, 58, 0.18) !important;
    border-radius: 26px !important;
    padding: 1.4rem !important;
    box-shadow: 0 22px 70px rgba(16, 33, 24, 0.10) !important;
}

/* Extra breathing space between fields */
[data-testid="stTextInput"] {
    margin-bottom: .95rem !important;
}

/* ---------------------------------------------------------
   LABELS
--------------------------------------------------------- */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: #102118 !important;
    font-weight: 850 !important;
    font-size: .92rem !important;
    margin-bottom: .38rem !important;
}

/* ---------------------------------------------------------
   INPUT WRAPPER
   This fixes both username and password borders properly.
--------------------------------------------------------- */
[data-testid="stTextInput"] div[data-baseweb="input"] {
    background: #ffffff !important;
    border: 1.8px solid #9fbea9 !important;
    border-radius: 15px !important;
    min-height: 52px !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: 0 8px 22px rgba(16, 33, 24, 0.045) !important;
    transition:
        border-color .16s ease,
        box-shadow .16s ease,
        background .16s ease !important;
}

/* Force every inner BaseWeb layer to stay white */
[data-testid="stTextInput"] div[data-baseweb="input"] > div,
[data-testid="stTextInput"] div[data-baseweb="input"] div {
    background: #ffffff !important;
}

/* Focus border */
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    border-color: #0f7a3a !important;
    box-shadow:
        0 0 0 3px rgba(15, 122, 58, 0.14),
        0 10px 24px rgba(16, 33, 24, 0.06) !important;
}

/* Actual input field */
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] input[type="text"],
[data-testid="stTextInput"] input[type="password"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #102118 !important;
    -webkit-text-fill-color: #102118 !important;
    caret-color: #0f7a3a !important;
    border: 0 !important;
    outline: 0 !important;
    min-height: 50px !important;
    font-size: .96rem !important;
    font-weight: 550 !important;
    padding-left: 1rem !important;
    padding-right: .7rem !important;
    box-shadow: none !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #829286 !important;
    -webkit-text-fill-color: #829286 !important;
    opacity: 1 !important;
}

/* Prevent browser autofill from turning inputs dark/yellow */
[data-testid="stTextInput"] input:-webkit-autofill,
[data-testid="stTextInput"] input:-webkit-autofill:hover,
[data-testid="stTextInput"] input:-webkit-autofill:focus,
[data-testid="stTextInput"] input:-webkit-autofill:active {
    -webkit-text-fill-color: #102118 !important;
    box-shadow: 0 0 0 1000px #ffffff inset !important;
    transition: background-color 9999s ease-in-out 0s !important;
}

/* ---------------------------------------------------------
   PASSWORD EYE BUTTON
   Properly aligned and color-managed inside the same border.
--------------------------------------------------------- */
[data-testid="stTextInput"] div[data-baseweb="input"] button {
    width: 46px !important;
    height: 42px !important;
    min-width: 46px !important;
    margin: 0 5px 0 0 !important;
    padding: 0 !important;
    border-radius: 12px !important;
    background: #eef8f1 !important;
    border: 1px solid #c9dfd0 !important;
    color: #0f7a3a !important;
    box-shadow: none !important;
    display: grid !important;
    place-items: center !important;
}

[data-testid="stTextInput"] div[data-baseweb="input"] button:hover {
    background: #dff2e6 !important;
    border-color: #0f7a3a !important;
    color: #064e2f !important;
}

/* Eye icon itself */
[data-testid="stTextInput"] div[data-baseweb="input"] button svg {
    color: #0f7a3a !important;
    fill: #0f7a3a !important;
    stroke: #0f7a3a !important;
}

/* ---------------------------------------------------------
   TEXTAREA
--------------------------------------------------------- */
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #102118 !important;
    border: 1.8px solid #9fbea9 !important;
    border-radius: 15px !important;
    font-size: .96rem !important;
    box-shadow: 0 8px 22px rgba(16, 33, 24, 0.045) !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #0f7a3a !important;
    box-shadow: 0 0 0 3px rgba(15, 122, 58, 0.14) !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #829286 !important;
    opacity: 1 !important;
}

/* ---------------------------------------------------------
   PRIMARY BUTTON
--------------------------------------------------------- */
[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    min-height: 52px !important;
    border-radius: 15px !important;
    border: 1px solid #0b6b35 !important;
    color: #ffffff !important;
    font-weight: 950 !important;
    font-size: .98rem !important;
    background: linear-gradient(135deg, #0b6b35, #19a957) !important;
    box-shadow: 0 16px 34px rgba(15, 122, 58, 0.24) !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
}

/* ---------------------------------------------------------
   SECONDARY NAV BUTTONS
--------------------------------------------------------- */
.stButton button {
    min-height: 44px !important;
    border-radius: 14px !important;
    border: 1px solid #b9d2c0 !important;
    color: #0b6b35 !important;
    background: #ffffff !important;
    font-weight: 900 !important;
    box-shadow: 0 10px 28px rgba(16, 33, 24, 0.07) !important;
}

.stButton button:hover {
    background: #e8f7ee !important;
    border-color: #0f7a3a !important;
}

/* ---------------------------------------------------------
   DIVIDER + FOOTER
--------------------------------------------------------- */
.divider-text {
    display: flex;
    align-items: center;
    gap: .8rem;
    margin: 1.05rem 0 .75rem 0 !important;
    color: #647067 !important;
    font-size: .82rem !important;
    font-weight: 850 !important;
}

.divider-text::before,
.divider-text::after {
    content: "";
    height: 1px;
    flex: 1;
    background: #dce8df;
}

.auth-footer {
    color: #647067 !important;
    font-size: .78rem !important;
    margin-top: 1rem !important;
    text-align: center;
}

.auth-hint {
    margin-top: .85rem;
    padding: .9rem 1rem;
    border-radius: 16px;
    background: #e8f7ee;
    border: 1px solid #cae8d3;
    color: #0f7a3a;
    font-size: .86rem;
    font-weight: 800;
    line-height: 1.55;
}

/* ---------------------------------------------------------
   RESPONSIVE
--------------------------------------------------------- */
@media (max-width: 950px) {
    .auth-layout {
        grid-template-columns: 1fr;
    }

    .brand-panel {
        min-height: 410px;
    }
}
</style>
"""
LOGIN_ANIMATION_CSS = """
<style>
/* ---------------------------------------------------------
   LOGIN PAGE ANIMATIONS ONLY
   Does not change original colors or authentication code.
--------------------------------------------------------- */

.auth-layout {
    animation: authPageEnter .75s cubic-bezier(.22, 1, .36, 1) both;
}

@keyframes authPageEnter {
    from {
        opacity: 0;
        transform: translateY(18px);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
        filter: blur(0);
    }
}

.brand-panel {
    animation:
        brandPanelEnter .85s cubic-bezier(.22, 1, .36, 1) both,
        brandPanelGlow 7s ease-in-out infinite 1s;
}

@keyframes brandPanelEnter {
    from {
        opacity: 0;
        transform: translateX(-26px) scale(.985);
        filter: blur(5px);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
        filter: blur(0);
    }
}

@keyframes brandPanelGlow {
    0%, 100% {
        box-shadow: 0 30px 90px rgba(6, 78, 47, 0.25);
    }
    50% {
        box-shadow: 0 36px 105px rgba(6, 78, 47, 0.33);
    }
}

.brand-pill {
    animation: authFadeDown .7s cubic-bezier(.22, 1, .36, 1) both .15s;
}

.brand-title {
    animation: authTitleReveal .85s cubic-bezier(.22, 1, .36, 1) both .25s;
}

.brand-subtitle {
    animation: authFadeUp .75s cubic-bezier(.22, 1, .36, 1) both .38s;
}

.brand-footer {
    animation: authFadeUp .75s cubic-bezier(.22, 1, .36, 1) both .52s;
}

@keyframes authTitleReveal {
    from {
        opacity: 0;
        transform: translateY(24px);
        letter-spacing: -.11em;
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
        letter-spacing: -.075em;
        filter: blur(0);
    }
}

@keyframes authFadeDown {
    from {
        opacity: 0;
        transform: translateY(-14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes authFadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.auth-header {
    animation: authRightEnter .72s cubic-bezier(.22, 1, .36, 1) both .20s;
}

[data-testid="stForm"] {
    animation: authFormEnter .78s cubic-bezier(.22, 1, .36, 1) both .30s;
}

@keyframes authRightEnter {
    from {
        opacity: 0;
        transform: translateX(22px);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
        filter: blur(0);
    }
}

@keyframes authFormEnter {
    from {
        opacity: 0;
        transform: translateY(22px) scale(.985);
        filter: blur(5px);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
}

.brand-dot {
    animation: brandDotPulse 1.8s ease-out infinite;
}

@keyframes brandDotPulse {
    0% {
        box-shadow: 0 0 0 0 rgba(255,255,255,.46);
        transform: scale(1);
    }
    70% {
        box-shadow: 0 0 0 12px rgba(255,255,255,0);
        transform: scale(1.08);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(255,255,255,0);
        transform: scale(1);
    }
}

[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextArea"] textarea {
    transition:
        border-color .18s ease,
        box-shadow .18s ease,
        transform .18s ease,
        background .18s ease !important;
}

[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
[data-testid="stTextArea"] textarea:hover {
    transform: translateY(-1px);
    border-color: #0f7a3a !important;
    box-shadow: 0 12px 28px rgba(16, 33, 24, 0.075) !important;
}

[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    transform: translateY(-1px);
}

[data-testid="stFormSubmitButton"] button,
.stButton button {
    transition:
        transform .18s ease,
        filter .18s ease,
        box-shadow .18s ease,
        border-color .18s ease,
        background .18s ease !important;
}

[data-testid="stFormSubmitButton"] button:hover,
.stButton button:hover {
    transform: translateY(-2px);
}

[data-testid="stFormSubmitButton"] button:active,
.stButton button:active {
    transform: translateY(0) scale(.99);
}

.divider-text,
.auth-footer,
.auth-hint {
    animation: authFadeUp .72s cubic-bezier(.22, 1, .36, 1) both .44s;
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
        scroll-behavior: auto !important;
    }
}
</style>
"""
def _ensure_auth_dir() -> None:
    LOCAL_AUTH_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    _ensure_auth_dir()
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data: Any) -> None:
    _ensure_auth_dir()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _hash_password(password: str, salt_b64: str | None = None) -> dict[str, Any]:
    if salt_b64:
        salt = base64.b64decode(salt_b64.encode("utf-8"))
    else:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )

    return {
        "algorithm": "pbkdf2_sha256",
        "iterations": PBKDF2_ITERATIONS,
        "salt": base64.b64encode(salt).decode("utf-8"),
        "hash": base64.b64encode(password_hash).decode("utf-8"),
    }


def _verify_password(password: str, password_record: dict[str, Any]) -> bool:
    try:
        current = _hash_password(password, str(password_record["salt"]))
        return hmac.compare_digest(str(current["hash"]), str(password_record["hash"]))
    except Exception:
        return False


def _load_local_users() -> list[dict[str, Any]]:
    return _load_json(LOCAL_USERS_FILE, [])


def _save_local_users(users: list[dict[str, Any]]) -> None:
    _save_json(LOCAL_USERS_FILE, users)


def _find_local_user_by_username(username: str) -> dict[str, Any] | None:
    username = username.strip().lower()
    for user in _load_local_users():
        if str(user.get("username", "")).strip().lower() == username:
            return user
    return None


def _find_local_user_by_email(email: str) -> dict[str, Any] | None:
    email = email.strip().lower()
    for user in _load_local_users():
        if str(user.get("email", "")).strip().lower() == email:
            return user
    return None


def _password_is_valid(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(ch.isalpha() for ch in password):
        return False, "Password must contain at least one letter."
    if not any(ch.isdigit() for ch in password):
        return False, "Password must contain at least one number."
    return True, "OK"


def _create_local_user(full_name: str, email: str, username: str, password: str) -> tuple[bool, str]:
    username_clean = username.strip().lower()
    email_clean = email.strip().lower()

    if _find_local_user_by_username(username_clean):
        return False, "Username already exists."

    if _find_local_user_by_email(email_clean):
        return False, "Email already exists."

    users = _load_local_users()
    users.append(
        {
            "user_id": f"local-{secrets.token_hex(8)}",
            "full_name": full_name.strip(),
            "email": email_clean,
            "username": username_clean,
            "role": "analyst",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "password": _hash_password(password),
        }
    )
    _save_local_users(users)
    return True, "Account created successfully. You can login now."


def _authenticate_local_user(username: str, password: str) -> dict[str, Any] | None:
    user = _find_local_user_by_username(username)
    if not user:
        return None
    if str(user.get("status", "active")) != "active":
        return None
    if _verify_password(password, user.get("password", {})):
        return user
    return None


def _load_reset_tokens() -> list[dict[str, Any]]:
    return _load_json(RESET_TOKENS_FILE, [])


def _save_reset_tokens(tokens: list[dict[str, Any]]) -> None:
    _save_json(RESET_TOKENS_FILE, tokens)


def _create_reset_token(email: str) -> str | None:
    user = _find_local_user_by_email(email)
    if not user:
        return None

    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_MINUTES)

    tokens = _load_reset_tokens()
    tokens.append(
        {
            "token": token,
            "email": str(user["email"]).lower(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at.isoformat(),
            "used": False,
        }
    )
    _save_reset_tokens(tokens)
    return token


def _reset_password_with_token(token: str, new_password: str) -> tuple[bool, str]:
    tokens = _load_reset_tokens()
    token_clean = token.strip()

    matched = None
    for item in tokens:
        if hmac.compare_digest(str(item.get("token", "")), token_clean):
            matched = item
            break

    if not matched:
        return False, "Invalid reset token."

    if matched.get("used"):
        return False, "This reset token has already been used."

    try:
        expires_at = datetime.fromisoformat(str(matched["expires_at"]))
    except Exception:
        return False, "Invalid token expiry."

    if datetime.now(timezone.utc) > expires_at:
        return False, "Reset token has expired."

    users = _load_local_users()
    email = str(matched["email"]).lower()
    updated = False

    for user in users:
        if str(user.get("email", "")).lower() == email:
            user["password"] = _hash_password(new_password)
            user["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = True
            break

    if not updated:
        return False, "User account not found."

    matched["used"] = True
    matched["used_at"] = datetime.now(timezone.utc).isoformat()

    _save_local_users(users)
    _save_reset_tokens(tokens)
    return True, "Password reset successfully. You can login now."


def _smtp_configured() -> bool:
    return bool(
        os.getenv("ECHOSAFE_SMTP_HOST")
        and os.getenv("ECHOSAFE_SMTP_PORT")
        and os.getenv("ECHOSAFE_SMTP_USER")
        and os.getenv("ECHOSAFE_SMTP_PASSWORD")
    )


def _send_reset_email(email: str, token: str) -> bool:
    if not _smtp_configured():
        return False

    host = os.getenv("ECHOSAFE_SMTP_HOST", "")
    port = int(os.getenv("ECHOSAFE_SMTP_PORT", "587"))
    user = os.getenv("ECHOSAFE_SMTP_USER", "")
    password = os.getenv("ECHOSAFE_SMTP_PASSWORD", "")
    sender = os.getenv("ECHOSAFE_SMTP_FROM", user)
    app_url = os.getenv("ECHOSAFE_APP_URL", "http://localhost:8501")
    reset_link = f"{app_url}?reset_token={token}"

    msg = EmailMessage()
    msg["Subject"] = "EchoSafe Password Reset"
    msg["From"] = sender
    msg["To"] = email
    msg.set_content(
        f"""EchoSafe password reset

Use this reset link:
{reset_link}

Or use this reset token:
{token}

This token expires in {RESET_TOKEN_MINUTES} minutes.
"""
    )

    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        return True
    except Exception as exc:
        logger.error("Failed to send reset email: %s", exc)
        return False


def _get_query_reset_token() -> str:
    try:
        value = st.query_params.get("reset_token", "")
        if isinstance(value, list):
            return value[0] if value else ""
        return str(value or "")
    except Exception:
        return ""


def _set_mode(mode: str) -> None:
    st.session_state["auth_mode"] = mode


def render_login_page() -> None:
    """Render clean professional login page."""
    st.markdown(LOGIN_CSS + LOGIN_ANIMATION_CSS, unsafe_allow_html=True)

    query_token = _get_query_reset_token()
    if query_token:
        st.session_state["auth_mode"] = "reset"
        st.session_state["reset_prefill_token"] = query_token

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    left, right = st.columns([1.05, 0.82], gap="large")

    with left:
        st.markdown(
            """
            <div class="brand-panel">
                <div class="brand-content">
                    <div class="brand-pill">
                        <span class="brand-dot"></span>
                        Live Risk Awareness
                    </div>
                    <h1 class="brand-title">EchoSafe</h1>
                    <p class="brand-subtitle">
                        A clean disaster-risk dashboard for weather signals,
                        monitored regions, active alerts, and earthquake event awareness.
                    </p>
                </div>
                <div class="brand-footer">
                    Secure access for disaster monitoring, early warning awareness,
                    and operational dashboard review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        mode = st.session_state.get("auth_mode", "login")

        if mode == "login":
            _render_login_screen()
        elif mode == "signup":
            _render_signup_screen()
        elif mode == "forgot":
            _render_forgot_screen()
        elif mode == "reset":
            _render_reset_screen()

        st.markdown(
            """
            <div class="auth-footer">
                EchoSafe secure dashboard access
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_panel_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="auth-header">
            <div class="auth-logo">🛡️</div>
            <h2 class="auth-title">{title}</h2>
            <p class="auth-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_login_screen() -> None:
    _render_panel_header("Login to EchoSafe", "Enter your credentials to continue.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username",
            placeholder="Enter username",
            key="login_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            key="login_password",
        )

        submit = st.form_submit_button("Login", use_container_width=True)

    if submit:
        username_clean = username.strip()
        if not username_clean or not password:
            st.error("Please enter both username and password.")
        elif authenticate_local(username_clean, password):
            logger.info("Successful login for user=%s", username_clean)
            st.session_state["_echosafe_login_success_redirect"] = True
            st.session_state["active_page"] = "Home"
            st.rerun()
        else:
            logger.warning("Failed login attempt for user=%s", username_clean)
            st.error("Invalid username or password.")

    st.markdown('<div class="divider-text">Need access?</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create Account", use_container_width=True):
            _set_mode("signup")
            st.rerun()
    with c2:
        if st.button("Forgot Password", use_container_width=True):
            _set_mode("forgot")
            st.rerun()


def _render_signup_screen() -> None:
    _render_panel_header(
        "Create Account",
        "Create your account and use the same credentials to login.",
    )

    with st.form("signup_form", clear_on_submit=False):
        full_name = st.text_input(
            "Full Name",
            placeholder="Enter full name",
            key="signup_full_name",
        )

        email = st.text_input(
            "Email",
            placeholder="name@example.com",
            key="signup_email",
        )

        username = st.text_input(
            "Username",
            placeholder="Choose username",
            key="signup_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Create password",
            key="signup_password",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Confirm password",
            key="signup_confirm_password",
        )

        submit = st.form_submit_button("Create Account", use_container_width=True)

    if submit:
        full_name_clean = full_name.strip()
        email_clean = email.strip().lower()
        username_clean = username.strip().lower()

        if not full_name_clean or not email_clean or not username_clean:
            st.error("Full name, email, and username are required.")
            return

        if "@" not in email_clean or "." not in email_clean:
            st.error("Please enter a valid email address.")
            return

        if password != confirm_password:
            st.error("Password and confirm password do not match.")
            return

        valid, msg = _password_is_valid(password)
        if not valid:
            st.error(msg)
            return

        ok, message = _create_local_user(
            full_name_clean,
            email_clean,
            username_clean,
            password,
        )

        if ok:
            st.success(message)
            st.session_state["auth_mode"] = "login"
            st.info("Your account is ready. Go back to login and sign in.")
        else:
            st.error(message)

    st.markdown('<div class="divider-text">Already have an account?</div>', unsafe_allow_html=True)

    if st.button("Back to Login", use_container_width=True):
        _set_mode("login")
        st.rerun()


def _render_forgot_screen() -> None:
    _render_panel_header(
        "Forgot Password",
        "Enter your registered email to receive a reset link.",
    )

    with st.form("forgot_form", clear_on_submit=False):
        email = st.text_input(
            "Email",
            placeholder="Enter registered email",
            key="forgot_email",
        )

        submit = st.form_submit_button("Send Reset Link", use_container_width=True)

    if submit:
        email_clean = email.strip().lower()

        if not email_clean:
            st.error("Please enter your email address.")
            return

        token = _create_reset_token(email_clean)

        if token:
            sent = _send_reset_email(email_clean, token)

            if sent:
                st.success("Password reset link has been sent to your email.")
            else:
                st.success("Reset request created.")
                st.warning(
                    "Email service is not configured on this machine. "
                    "Use this local reset token for testing."
                )
                st.code(token)

                if st.button("Reset Password Now", use_container_width=True):
                    st.session_state["reset_prefill_token"] = token
                    _set_mode("reset")
                    st.rerun()
        else:
            st.success("If an account exists with this email, a reset link will be sent.")

    st.markdown('<div class="divider-text">Remembered your password?</div>', unsafe_allow_html=True)

    if st.button("Back to Login", use_container_width=True):
        _set_mode("login")
        st.rerun()


def _render_reset_screen() -> None:
    _render_panel_header(
        "Reset Password",
        "Paste your reset token and create a new password.",
    )

    prefill = st.session_state.get("reset_prefill_token", "")

    with st.form("reset_form", clear_on_submit=False):
        token = st.text_input(
            "Reset Token",
            value=prefill,
            placeholder="Paste reset token",
            key="reset_token",
        )

        new_password = st.text_input(
            "New Password",
            type="password",
            placeholder="Enter new password",
            key="reset_new_password",
        )

        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
            placeholder="Confirm new password",
            key="reset_confirm_password",
        )

        submit = st.form_submit_button("Reset Password", use_container_width=True)

    if submit:
        token_clean = token.strip()

        if not token_clean:
            st.error("Reset token is required.")
            return

        if new_password != confirm_password:
            st.error("New password and confirm password do not match.")
            return

        valid, msg = _password_is_valid(new_password)
        if not valid:
            st.error(msg)
            return

        ok, message = _reset_password_with_token(token_clean, new_password)

        if ok:
            st.success(message)
            st.session_state.pop("reset_prefill_token", None)
            st.session_state["auth_mode"] = "login"
            st.info("Password updated. Go back to login and sign in.")
        else:
            st.error(message)

    st.markdown('<div class="divider-text">Return</div>', unsafe_allow_html=True)

    if st.button("Back to Login", use_container_width=True):
        _set_mode("login")
        st.rerun()


def authenticate_local(username: str, password: str) -> bool:
    """
    Authenticate using:
    1. Existing project security layer.
    2. Local users created from this login page.

    This does not change pipeline, ML model, prediction, or dashboard page logic.
    """
    from backend.config.security import JWTManager, authenticate_user

    user = authenticate_user(username, password)

    if user:
        token = JWTManager.create_access_token(
            user["user_id"],
            user["username"],
            user["role"],
        )
        login_user(username, token)
        return True

    local_user = _authenticate_local_user(username, password)

    if local_user:
        token = JWTManager.create_access_token(
            str(local_user["user_id"]),
            str(local_user["username"]),
            str(local_user.get("role", "analyst")),
        )
        login_user(str(local_user["username"]), token)
        return True

    return False