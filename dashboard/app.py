#!/usr/bin/env python3
"""EchoSafe Streamlit Dashboard.

Professional custom-routed Streamlit dashboard.

Routing behavior:
- Base URL http://localhost:8501 shows Login.
- Dashboard pages use safe URL routing: ?page=home, ?page=heatwave, etc.
- JWT is NOT stored in URL.
- Browser refresh on a valid dashboard page keeps that page.
- Browser Back / Forward works between dashboard pages.
- Back from Home to bare URL returns to Login.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.auth.session import init_session, is_authenticated, logout, get_current_user

try:
    from dashboard.auth.session import (
        restore_session_from_cookie,
        persist_auth_cookie,
        clear_auth_cookie_if_requested,
    )
except Exception:
    def restore_session_from_cookie() -> bool:
        return False

    def persist_auth_cookie() -> None:
        return None

    def clear_auth_cookie_if_requested() -> None:
        return None

from dashboard.auth.login import render_login_page
from dashboard.pages.home import render_home
from dashboard.pages.flood import render_flood_page
from dashboard.pages.heatwave import render_heatwave_page
from dashboard.pages.earthquake import render_earthquake_page
from dashboard.pages.hailstorm import render_hailstorm_page
from dashboard.pages.regional_overview import render_regional_overview
from dashboard.pages.alerts import render_alerts_page
from dashboard.utils.dashboard_ui import inject_global_css


st.set_page_config(
    page_title="EchoSafe | Disaster Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "EchoSafe Disaster Management and Forecasting System",
    },
)

try:
    st.set_option("client.showSidebarNavigation", False)
    st.set_option("client.toolbarMode", "viewer")
except Exception:
    pass


PAGES = {
    "Home": {"icon": "☀️", "fn": render_home},
    "Heatwave Risk": {"icon": "🌡️", "fn": render_heatwave_page},
    "Hailstorm Risk": {"icon": "⛈️", "fn": render_hailstorm_page},
    "Earthquake Monitoring": {"icon": "🌍", "fn": render_earthquake_page},
    "Flood / Heavy Rainfall": {"icon": "🌊", "fn": render_flood_page},
    "Regional Overview": {"icon": "🗺️", "fn": render_regional_overview},
    "Alerts": {"icon": "🚨", "fn": render_alerts_page},
}

PAGE_TO_SLUG = {
    "Home": "home",
    "Heatwave Risk": "heatwave",
    "Hailstorm Risk": "hailstorm",
    "Earthquake Monitoring": "earthquake",
    "Flood / Heavy Rainfall": "flood",
    "Regional Overview": "regional-overview",
    "Alerts": "alerts",
}

SLUG_TO_PAGE = {slug: page for page, slug in PAGE_TO_SLUG.items()}

PAGE_QUERY_KEY = "page"
HAS_ENTERED_DASHBOARD_KEY = "_echosafe_has_entered_dashboard"
LAST_RENDERED_PAGE_KEY = "_echosafe_last_rendered_page"

OLD_PAGE_COOKIE_NAME = "echosafe_active_page"


def _query_get(key: str) -> str | None:
    try:
        value = st.query_params.get(key)
        if isinstance(value, list):
            return value[0] if value else None
        return str(value) if value is not None else None
    except Exception:
        try:
            value = st.experimental_get_query_params().get(key)
            if isinstance(value, list):
                return value[0] if value else None
            return str(value) if value is not None else None
        except Exception:
            return None


def _query_set(key: str, value: str) -> None:
    try:
        if _query_get(key) != value:
            st.query_params[key] = value
    except Exception:
        try:
            params = st.experimental_get_query_params()
            current = params.get(key, [None])
            current_value = current[0] if isinstance(current, list) and current else current

            if current_value != value:
                params[key] = value
                st.experimental_set_query_params(**params)
        except Exception:
            pass


def _query_delete(key: str) -> None:
    try:
        if key in st.query_params:
            del st.query_params[key]
    except Exception:
        try:
            params = st.experimental_get_query_params()
            if key in params:
                params.pop(key, None)
                st.experimental_set_query_params(**params)
        except Exception:
            pass


def _delete_browser_cookie(name: str) -> None:
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


def _clear_old_unsafe_query_params() -> None:
    """Remove old unsafe auth query params. Keep safe ?page=... routing."""
    unsafe_keys = [
        "echosafe_token",
        "access_token",
        "auth_token",
        "jwt",
        "token",
    ]

    try:
        for key in unsafe_keys:
            if key in st.query_params:
                del st.query_params[key]
    except Exception:
        try:
            params = st.experimental_get_query_params()
            changed = False

            for key in unsafe_keys:
                if key in params:
                    params.pop(key, None)
                    changed = True

            if changed:
                st.experimental_set_query_params(**params)
        except Exception:
            pass


def _inject_navigation_css() -> None:
    st.markdown(
        """
        <style>
        .sidebar-nav-button {
            margin-bottom: .28rem;
        }

        .active-page-label {
            padding: .58rem .75rem;
            margin: .12rem 0 .35rem 0;
            border-radius: 14px;
            color: #ffffff;
            font-weight: 900;
            font-size: .92rem;
            background: linear-gradient(135deg, rgba(15, 122, 58, 0.95), rgba(25, 169, 87, 0.95));
            border: 1px solid rgba(34, 197, 94, 0.70);
            box-shadow: 0 12px 28px rgba(15, 122, 58, 0.26);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _go_to_dashboard_home_after_login() -> None:
    """Move into dashboard route after successful login."""
    st.session_state.active_page = "Home"
    st.session_state[HAS_ENTERED_DASHBOARD_KEY] = True
    st.session_state[LAST_RENDERED_PAGE_KEY] = "Home"
    _query_set(PAGE_QUERY_KEY, "home")
    st.rerun()


def _render_login_and_stop(clear_auth: bool = False) -> None:
    """Render login page and stop execution."""
    _delete_browser_cookie(OLD_PAGE_COOKIE_NAME)

    if clear_auth:
        logout()
        clear_auth_cookie_if_requested()
        st.session_state[HAS_ENTERED_DASHBOARD_KEY] = False
        st.session_state[LAST_RENDERED_PAGE_KEY] = None
        st.session_state.active_page = "Home"

    render_login_page()
    st.stop()


def _resolve_active_page() -> None:
    """Resolve current page without fighting URL navigation.

    Priority:
    1. Internal page button changed st.session_state.active_page.
    2. Browser URL ?page=slug.
    3. Home fallback.
    """
    query_slug = _query_get(PAGE_QUERY_KEY)
    query_page = SLUG_TO_PAGE.get(query_slug) if query_slug else None

    session_page = st.session_state.get("active_page")
    last_rendered_page = st.session_state.get(LAST_RENDERED_PAGE_KEY)

    # First dashboard render or browser refresh/direct URL:
    # URL must win.
    if last_rendered_page is None and query_page:
        active_page = query_page

    # Internal app navigation:
    # Your page buttons set st.session_state.active_page then call st.rerun().
    elif session_page in PAGES and session_page != last_rendered_page:
        active_page = session_page

    # Browser Back / Forward:
    # session_page is same as last rendered, but URL has changed.
    elif query_page:
        active_page = query_page

    # Safe fallback.
    else:
        active_page = "Home"

    if active_page not in PAGES:
        active_page = "Home"

    active_slug = PAGE_TO_SLUG.get(active_page, "home")

    st.session_state.active_page = active_page
    st.session_state[LAST_RENDERED_PAGE_KEY] = active_page
    st.session_state[HAS_ENTERED_DASHBOARD_KEY] = True

    if _query_get(PAGE_QUERY_KEY) != active_slug:
        _query_set(PAGE_QUERY_KEY, active_slug)


def _navigate_to(page_name: str) -> None:
    """Sidebar navigation request."""
    if page_name not in PAGES:
        page_name = "Home"

    st.session_state.active_page = page_name
    st.rerun()


inject_global_css()
_inject_navigation_css()
init_session()

_clear_old_unsafe_query_params()
clear_auth_cookie_if_requested()

current_page_slug = _query_get(PAGE_QUERY_KEY)
current_page_is_valid = current_page_slug in SLUG_TO_PAGE if current_page_slug else False

# Bare URL = Login.
# After successful login, redirect once to ?page=home.
if not current_page_is_valid:
    login_success_redirect = st.session_state.pop("_echosafe_login_success_redirect", False)

    if login_success_redirect and is_authenticated():
        _go_to_dashboard_home_after_login()

    if is_authenticated() and not st.session_state.get(HAS_ENTERED_DASHBOARD_KEY):
        _go_to_dashboard_home_after_login()

    _render_login_and_stop(clear_auth=is_authenticated())

# Valid dashboard page URL may restore auth from cookie.
restore_session_from_cookie()

if not is_authenticated():
    render_login_page()
    st.stop()

persist_auth_cookie()

_resolve_active_page()

with st.sidebar:
    user = get_current_user() or {}
    username = user.get("username", "User")
    role = str(user.get("role", "analyst")).title()

    st.markdown(
        f"""
        <div class="sidebar-logo-card">
            <div style="font-size:2.25rem;line-height:1;">🛡️</div>
            <div class="sidebar-title">EchoSafe</div>
            <div class="sidebar-sub">Disaster risk monitoring for selected Pakistan regions.</div>
        </div>
        <div style="color:#94a3b8;font-size:.78rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin:.3rem 0 .6rem;">
            Navigation
        </div>
        """,
        unsafe_allow_html=True,
    )

    for label, meta in PAGES.items():
        active = st.session_state.active_page == label

        if active:
            st.markdown(
                f"""
                <div class="active-page-label">
                    {meta["icon"]}&nbsp;&nbsp;{label}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="sidebar-nav-button">', unsafe_allow_html=True)

            if st.button(
                f"{meta['icon']}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
            ):
                _navigate_to(label)

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        f"""
        <div class="glass-card" style="padding:.9rem;border-radius:18px;">
            <div style="color:#94a3b8;font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;">Signed in</div>
            <div style="font-weight:900;font-size:1.05rem;margin-top:.2rem;">{username}</div>
            <div style="color:#cbd5e1;font-size:.84rem;">{role}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Logout", use_container_width=True):
        logout()
        clear_auth_cookie_if_requested()
        st.session_state[HAS_ENTERED_DASHBOARD_KEY] = False
        st.session_state[LAST_RENDERED_PAGE_KEY] = None
        st.session_state.active_page = "Home"
        _query_delete(PAGE_QUERY_KEY)
        st.rerun()

PAGES[st.session_state.active_page]["fn"]()