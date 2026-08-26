"""Small demo-only authentication helpers for the Streamlit app."""

from __future__ import annotations

import hashlib
import hmac


def _password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# These accounts are intentionally simple and are only suitable for demos.
DEMO_USERS = {
    "demo": {
        "password_hash": _password_hash("demo123"),
        "display_name": "演示用户",
    },
    "admin": {
        "password_hash": _password_hash("admin123"),
        "display_name": "演示管理员",
    },
}


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    """Return the user record when demo credentials are valid."""
    user = DEMO_USERS.get(username.strip())
    if user is None:
        return None
    if not hmac.compare_digest(user["password_hash"], _password_hash(password)):
        return None
    return {"user_id": username.strip(), "display_name": user["display_name"]}
