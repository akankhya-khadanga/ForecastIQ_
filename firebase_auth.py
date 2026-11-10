"""Firebase Authentication REST helpers for the ForecastIQ Streamlit app."""
from __future__ import annotations

import requests

IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1"
SECURE_TOKEN = "https://securetoken.googleapis.com/v1/token"


class FirebaseAuthError(Exception):
    """Readable authentication error returned by Firebase."""


def _message(response: requests.Response) -> str:
    try:
        raw = response.json().get("error", {}).get("message", "Authentication failed")
    except Exception:
        raw = "Authentication failed"
    messages = {
        "EMAIL_EXISTS": "An account with this email already exists.",
        "OPERATION_NOT_ALLOWED": "Email/password sign-in is not enabled in Firebase Authentication.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please wait and try again.",
        "EMAIL_NOT_FOUND": "No account exists with this email.",
        "INVALID_PASSWORD": "The password is incorrect.",
        "INVALID_LOGIN_CREDENTIALS": "The email or password is incorrect.",
        "USER_DISABLED": "This account has been disabled.",
        "WEAK_PASSWORD : Password should be at least 6 characters": "Use a password with at least 6 characters.",
        "INVALID_EMAIL": "Enter a valid email address.",
    }
    return messages.get(raw, raw.replace("_", " ").capitalize())


def sign_up(api_key: str, email: str, password: str) -> dict:
    response = requests.post(
        f"{IDENTITY_TOOLKIT}/accounts:signUp?key={api_key}",
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=20,
    )
    if not response.ok:
        raise FirebaseAuthError(_message(response))
    return response.json()


def sign_in(api_key: str, email: str, password: str) -> dict:
    response = requests.post(
        f"{IDENTITY_TOOLKIT}/accounts:signInWithPassword?key={api_key}",
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=20,
    )
    if not response.ok:
        raise FirebaseAuthError(_message(response))
    return response.json()


def refresh_session(api_key: str, refresh_token: str) -> dict:
    response = requests.post(
        f"{SECURE_TOKEN}?key={api_key}",
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        timeout=20,
    )
    if not response.ok:
        raise FirebaseAuthError(_message(response))
    return response.json()
