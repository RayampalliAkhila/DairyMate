"""
Access control.

Two ways in:

* **Google** — Streamlit's built-in OpenID Connect. Configure `[auth]` in
  `.streamlit/secrets.toml` and the button appears on its own. The identity
  cookie survives a browser refresh, which is what you want day to day.
* **Local passcode** — a username and password checked against PBKDF2 hashes
  in `[local_users]`. There so the app is usable before anyone has set up a
  Google Cloud project, and as a fallback when the network is down in a
  parlour office. It lives in session state only, so a refresh signs you out.

Either way an allowlist can narrow who gets in — `[access] emails` and
`[access] domains`. With no allowlist configured, any account your provider
authenticates is accepted, which is rarely what you want in production.

Generate a password hash:

    python core/auth.py "the password"
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets as pysecrets
import sys

import streamlit as st

_ITERATIONS = 260_000
_SESSION_KEY = "_dm::auth::user"


# ------------------------------------------------------------- passwords
def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _ITERATIONS,
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_b64, digest_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), base64.b64decode(salt_b64), int(iterations)
        )
        # Constant time — a timing leak here would let someone probe the hash.
        return hmac.compare_digest(digest, base64.b64decode(digest_b64))
    except Exception:      # noqa: BLE001
        return False


# ----------------------------------------------------------------- config
def _secret(section: str, default=None):
    try:
        return st.secrets[section]
    except Exception:      # noqa: BLE001
        return default


def google_configured() -> bool:
    auth = _secret("auth")
    if not auth:
        return False
    required = ("redirect_uri", "cookie_secret")
    has_shared = all(k in auth for k in required)
    has_client = "client_id" in auth or any(
        isinstance(v, dict) and "client_id" in v for v in auth.values()
    )
    return bool(has_shared and has_client and hasattr(st, "login"))


def local_users() -> dict:
    users = _secret("local_users", {})
    try:
        return dict(users)
    except Exception:      # noqa: BLE001
        return {}


def auth_enabled() -> bool:
    """No providers configured at all means the app runs open."""
    return google_configured() or bool(local_users())


def _allowed(email: str | None) -> bool:
    access = _secret("access", {}) or {}
    emails = [e.lower() for e in access.get("emails", [])]
    domains = [d.lower().lstrip("@") for d in access.get("domains", [])]
    if not emails and not domains:
        return True
    if not email:
        return False
    email = email.lower()
    return email in emails or email.rsplit("@", 1)[-1] in domains


# ------------------------------------------------------------------ state
def current_user() -> dict | None:
    """The signed-in user, from the Google cookie or the local session."""
    if google_configured():
        try:
            if st.user.is_logged_in:
                return {
                    "name": getattr(st.user, "name", None) or st.user.email,
                    "email": getattr(st.user, "email", None),
                    "method": "google",
                }
        except Exception:      # noqa: BLE001
            pass
    return st.session_state.get(_SESSION_KEY)


def sign_out() -> None:
    user = current_user()
    st.session_state.pop(_SESSION_KEY, None)
    if user and user.get("method") == "google":
        st.logout()
    else:
        st.rerun()


# --------------------------------------------------------------- login UI
_LOGIN_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarNav"],
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main .block-container { max-width: 480px; }
</style>
"""


def _login_screen(message: str | None = None) -> None:
    from core import theme

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    st.write("")

    st.markdown(
        f"""
        <div style="border-top:3px solid {theme.INK};padding-top:1rem;margin-bottom:1.4rem;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                      text-transform:uppercase;letter-spacing:0.16em;color:{theme.MUTED};">
            Dairy Mate
          </div>
          <div style="font-family:'Archivo',sans-serif;font-weight:700;font-size:1.9rem;
                      line-height:1.15;margin:0.15rem 0 0.35rem 0;color:{theme.INK};">
            Sign in
          </div>
          <div style="color:{theme.MUTED};font-size:0.93rem;">
            Mastitis screening console. Herd images and scores are clinical
            records — sign in with the account your holding registered.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if message:
        st.error(message)

    if google_configured():
        if st.button("Continue with Google", type="primary", use_container_width=True):
            st.login()

    users = local_users()
    if google_configured() and users:
        st.markdown(
            f"<div style='text-align:center;color:{theme.MUTED};font-size:0.78rem;"
            f"margin:0.9rem 0 0.4rem;'>or</div>",
            unsafe_allow_html=True,
        )

    if users:
        with st.form("dm_local_login", border=bool(google_configured())):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            stored = users.get(username.strip())
            if stored and verify_password(password, str(stored)):
                st.session_state[_SESSION_KEY] = {
                    "name": username.strip(),
                    "email": None,
                    "method": "local",
                }
                st.rerun()
            else:
                # One message for both cases — naming which half was wrong
                # tells an attacker which usernames exist.
                st.error("Those details were not recognised.")

    if not google_configured() and not users:
        st.info(
            "No sign-in method is configured yet. Add an `[auth]` or "
            "`[local_users]` section to `.streamlit/secrets.toml`, or remove "
            "the `require_login()` calls to run the app open."
        )

    st.markdown(
        f"<div style='margin-top:1.6rem;color:{theme.MUTED};font-size:0.76rem;'>"
        "Screening aid only. Every flagged animal needs a hands-on check before "
        "treatment.</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ gate
def require_login() -> dict | None:
    """
    Call once per page, straight after theme.inject().

    Renders the login screen and halts the page when nobody is signed in.
    Returns the user record otherwise.
    """
    if not auth_enabled():
        return None

    user = current_user()
    if user is None:
        _login_screen()
        st.stop()

    if not _allowed(user.get("email")):
        _login_screen(
            f"{user.get('email') or user.get('name')} is not on the access list "
            "for this console. Ask whoever administers it to add you."
        )
        if st.button("Sign out", use_container_width=True):
            sign_out()
        st.stop()

    return user


def sidebar_account() -> None:
    """User chip and sign-out, rendered inside the sidebar."""
    user = current_user()
    if user is None:
        return
    with st.sidebar:
        st.divider()
        label = user.get("email") or user.get("name")
        via = "Google" if user["method"] == "google" else "local account"
        st.markdown(f"**{user.get('name')}**")
        st.caption(f"{label} · {via}")
        if st.button("Sign out", key="dm_signout", use_container_width=True):
            sign_out()


# ------------------------------------------------------------------- cli
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python core/auth.py "the password"')
        print("Prints a line to paste under [local_users] in secrets.toml.")
        raise SystemExit(1)

    encoded = hash_password(sys.argv[1])
    assert verify_password(sys.argv[1], encoded), "self-check failed"

    print("Add to .streamlit/secrets.toml:\n")
    print("[local_users]")
    print(f'yourname = "{encoded}"')
    print(f"\nRandom cookie_secret if you need one:\n{pysecrets.token_hex(32)}")
