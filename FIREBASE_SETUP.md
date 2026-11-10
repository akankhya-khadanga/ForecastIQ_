# ForecastIQ — Firebase setup

This project includes email/password signup and login, logout, Firebase ID-token verification, Firestore user profiles, and an explicit **Save forecast to my account** action.

## 1. Create a Firebase project

1. Open <https://console.firebase.google.com/> and create a project (or select the project you want to use for ForecastIQ).
2. In **Build → Authentication → Get started → Sign-in method**, enable **Email/Password** and save.
3. In **Build → Firestore Database**, click **Create database**. Choose **Production mode**. Select a location close to your users; for an India-focused project, `asia-south1` (Mumbai) is a reasonable choice if available. Firestore's location cannot normally be changed after creation.
4. Open **Project settings → General**. Under **Your apps**, register a Web app if one does not already exist. Copy its **Web API Key**.
5. Open **Project settings → Service accounts → Firebase Admin SDK → Generate new private key**. Download the JSON file and keep it private.

## 2. Configure local secrets

1. In the project root, open the `.streamlit` folder.
2. Copy `secrets.toml.example` and name the copy `secrets.toml`.
3. Fill `[firebase].api_key` with the Web API Key.
4. Copy the matching fields from the downloaded service-account JSON into `[firebase_service_account]`: `type`, `project_id`, `private_key_id`, `private_key`, `client_email`, `client_id`, `auth_uri`, `token_uri`, `auth_provider_x509_cert_url`, `client_x509_cert_url`, and `universe_domain` (if present).
5. Preserve the private key's BEGIN/END lines. The `private_key` value must be a TOML multiline string (`"""..."""`).
6. Do not share `secrets.toml`, the service-account JSON, or commit either file to GitHub. `.gitignore` already excludes `.streamlit/secrets.toml`.

## 3. Install and run

From the project root:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run uploads/streamlit_app.py
```

The app will show Login and Create account tabs. A successful signup/login creates or updates a profile document at `users/{firebase_uid}`. The dashboard is not shown until the user is authenticated. The sidebar Logout button clears the Streamlit login session.

## 4. Firestore data written by this version

- `users/{firebase_uid}`: UID, email, display name (signup), created/updated timestamps, and last login timestamp.
- `forecast_runs/{run_id}`: user UID, model name, forecast horizon, date range, and creation timestamp. This is written only when the signed-in user clicks **Save forecast to my account**.
- `forecast_runs/{run_id}/results/{day_id}`: daily predicted sales and forecast interval bounds for that saved run.

The sales CSV remains the shared local dataset in this starter version. It is not automatically uploaded to Firestore. The app does not store passwords in Firestore; Firebase Authentication manages passwords.

## 5. Firestore security

The app uses the Firebase Admin SDK on the Python server. Admin SDK requests bypass Firestore client security rules, so keep the service-account key private and do not expose it in browser code or GitHub. The included `firestore.rules` denies direct client reads/writes; the trusted Python server accesses Firestore using the Admin SDK. If you later add direct browser/mobile Firestore access, replace these rules with carefully tested UID-based rules before enabling it.

## Troubleshooting

- **Firebase credentials are not configured yet:** check that the file is named exactly `.streamlit/secrets.toml` and is inside the project root's `.streamlit` directory.
- **Email/password sign-in is not enabled:** enable Email/Password under Firebase Authentication → Sign-in method.
- **Profile setup failed:** confirm Firestore Database has been created and all service-account values are copied correctly.
- **Permission or credential error:** verify that the service account belongs to the same Firebase project as the Web API key.
- **Module not found:** activate the same Python environment used to run Streamlit, then install `requirements.txt` again.
