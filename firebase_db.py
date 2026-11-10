
"""Server-side Firestore helpers.
Never expose the service-account key to browsers.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st


# ---------------------------------------------------------
# FIREBASE / FIRESTORE INITIALIZATION
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_firestore_client():
    """
    Initialize Firebase Admin and return Firestore client.

    Supports:
    1. Streamlit Community Cloud using st.secrets
    2. Local development using service-account JSON
    """

    try:
        # Reuse an already initialized Firebase application
        app = firebase_admin.get_app()

    except ValueError:

        # -------------------------------------------------
        # OPTION 1: STREAMLIT CLOUD SECRETS
        # -------------------------------------------------

        if "firebase_service_account" in st.secrets:

            service_account_info = dict(
                st.secrets["firebase_service_account"]
            )

            cred = credentials.Certificate(
                service_account_info
            )

        # -------------------------------------------------
        # OPTION 2: LOCAL DEVELOPMENT JSON FILE
        # -------------------------------------------------

        else:

            project_root = Path(__file__).resolve().parent

            service_account_path = (
                project_root
                / ".streamlit"
                / "firebase-service-account.json"
            )

            if not service_account_path.exists():
                raise FileNotFoundError(
                    "Firebase service-account credentials not found. "
                    "Configure firebase_service_account in Streamlit "
                    "Cloud Secrets or provide the local JSON file at: "
                    f"{service_account_path}"
                )

            cred = credentials.Certificate(
                str(service_account_path)
            )

        # Initialize Firebase Admin
        app = firebase_admin.initialize_app(cred)

    return firestore.client(app=app)


# ---------------------------------------------------------
# USER PROFILE MANAGEMENT
# ---------------------------------------------------------

def save_user_profile(
    uid: str,
    email: str,
    display_name: str = ""
):
    """Create or update the signed-in user's Firestore profile."""

    db = get_firestore_client()

    ref = db.collection("users").document(uid)

    existing = ref.get()

    now = datetime.now(timezone.utc)

    payload = {
        "uid": uid,
        "email": email.lower().strip(),
        "updated_at": now,
        "last_login_at": now,
    }

    if display_name.strip():
        payload["display_name"] = display_name.strip()

    if not existing.exists:
        payload["created_at"] = now

    ref.set(payload, merge=True)


def get_user_profile(uid: str):
    """Retrieve a user's profile."""

    return (
        get_firestore_client()
        .collection("users")
        .document(uid)
        .get()
    )


# ---------------------------------------------------------
# FORECAST HISTORY MANAGEMENT
# ---------------------------------------------------------

def save_forecast_run(
    uid: str,
    model_name: str,
    horizon_days: int,
    summary: dict | None = None,
) -> str:
    """Persist a forecast-run record owned by the authenticated user."""

    db = get_firestore_client()

    payload = {
        "user_id": uid,
        "model_name": model_name,
        "forecast_horizon_days": int(horizon_days),
        "created_at": datetime.now(timezone.utc),
        "summary": summary or {},
    }

    doc_ref = db.collection("forecast_runs").document()

    doc_ref.set(payload)

    return doc_ref.id


def save_forecast_to_firestore(
    uid: str,
    model_name: str,
    forecast_df,
) -> str:
    """Save one forecast run and its daily predictions."""

    db = get_firestore_client()

    rows = forecast_df.copy()

    if rows.empty:
        raise ValueError("Cannot save an empty forecast.")

    required_columns = {"date", "forecast"}

    missing_columns = required_columns - set(rows.columns)

    if missing_columns:
        raise ValueError(
            f"Forecast DataFrame is missing columns: {missing_columns}"
        )

    run_ref = db.collection("forecast_runs").document()

    dates = rows["date"].tolist()

    run_ref.set({
        "user_id": uid,
        "model_name": model_name,
        "forecast_horizon_days": int(len(rows)),
        "start_date": dates[0].date().isoformat(),
        "end_date": dates[-1].date().isoformat(),
        "created_at": datetime.now(timezone.utc),
    })

    batch = db.batch()

    for index, row in rows.reset_index(drop=True).iterrows():

        result_ref = (
            run_ref.collection("results")
            .document(f"day_{index + 1:03d}")
        )

        lower = row.get("lower")
        upper = row.get("upper")

        batch.set(result_ref, {
            "forecast_date": row["date"].date().isoformat(),
            "predicted_sales": float(row["forecast"]),
            "lower_bound": (
                float(lower) if lower is not None else None
            ),
            "upper_bound": (
                float(upper) if upper is not None else None
            ),
        })

    batch.commit()

    return run_ref.id
