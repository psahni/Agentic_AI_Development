import os
import base64
import json
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose"
]


def get_gmail_service():
    """Authenticate and return Gmail service."""
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def search_emails(query: str, max_results: int = 5) -> str:
    """Search Gmail inbox and return email summaries."""
    print(f"  📧 Searching emails: {query}")
    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results
    ).execute()

    messages = results.get("messages", [])
    if not messages:
        return "No emails found."

    emails = []
    for msg in messages:
        detail = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
        snippet = detail.get("snippet", "")

        emails.append(
            f"ID: {msg['id']}\n"
            f"From: {headers.get('From', 'Unknown')}\n"
            f"Subject: {headers.get('Subject', 'No subject')}\n"
            f"Date: {headers.get('Date', 'Unknown')}\n"
            f"Preview: {snippet}\n"
        )

    return "\n---\n".join(emails)


def read_email(email_id: str) -> str:
    """Read the full content of a specific email."""
    print(f"  📖 Reading email: {email_id}")
    service = get_gmail_service()

    msg = service.users().messages().get(
        userId="me",
        id=email_id,
        format="full"
    ).execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}

    # Extract body
    body = ""
    if "parts" in msg["payload"]:
        for part in msg["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                body = base64.urlsafe_b64decode(data).decode("utf-8")
                break
    elif "body" in msg["payload"]:
        data = msg["payload"]["body"].get("data", "")
        body = base64.urlsafe_b64decode(data).decode("utf-8")

    return (
        f"From: {headers.get('From', 'Unknown')}\n"
        f"Subject: {headers.get('Subject', 'No subject')}\n"
        f"Date: {headers.get('Date', 'Unknown')}\n\n"
        f"Body:\n{body[:2000]}"  # limit to 2000 chars
    )


def create_draft(to: str, subject: str, body: str) -> str:
    """Save a draft reply in Gmail."""
    print(f"  ✍️  Creating draft to: {to}")
    service = get_gmail_service()

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft = service.users().drafts().create(
        userId="me",
        body={"message": {"raw": encoded}}
    ).execute()

    return f"Draft saved successfully. Draft ID: {draft['id']}"