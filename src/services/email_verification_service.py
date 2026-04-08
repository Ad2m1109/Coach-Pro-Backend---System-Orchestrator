import logging
import os
import secrets
import smtplib
from email.message import EmailMessage
from typing import Optional

from env_loader import load_backend_env


load_backend_env()


logger = logging.getLogger(__name__)


def generate_email_verification_code(length: int = 6) -> str:
    digits = "0123456789"
    return "".join(secrets.choice(digits) for _ in range(length))


def _send_code_email(
    to_email: str,
    *,
    subject: str,
    intro_line: str,
    code: str,
    full_name: Optional[str] = None,
) -> None:
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_from_email = os.environ.get("SMTP_FROM_EMAIL", "").strip()
    dev_mode = os.environ.get("EMAIL_VERIFICATION_DEV_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    greeting_name = (full_name or "").strip() or "there"
    body = (
        f"Hello {greeting_name},\n\n"
        f"{intro_line}: {code}\n\n"
        "This code expires in 15 minutes.\n"
        "If you did not request this account, you can ignore this email.\n"
    )

    if not smtp_host or not smtp_from_email:
        if dev_mode:
            logger.warning(
                "EMAIL_VERIFICATION_DEV_MODE enabled. Verification code for %s is %s",
                to_email,
                code,
            )
            return
        raise RuntimeError(
            "Email delivery is not configured. Set SMTP_HOST and SMTP_FROM_EMAIL, "
            "or enable EMAIL_VERIFICATION_DEV_MODE for local development."
        )

    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_username = os.environ.get("SMTP_USERNAME", "").strip()
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_use_ssl = os.environ.get("SMTP_USE_SSL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    smtp_use_tls = os.environ.get("SMTP_USE_TLS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_from_email
    message["To"] = to_email
    message.set_content(body)

    if smtp_use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
            if smtp_username:
                server.login(smtp_username, smtp_password)
            server.send_message(message)
        return

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        if smtp_use_tls:
            server.starttls()
        if smtp_username:
            server.login(smtp_username, smtp_password)
        server.send_message(message)


def send_email_verification_email(
    to_email: str,
    code: str,
    *,
    full_name: Optional[str] = None,
) -> None:
    _send_code_email(
        to_email,
        subject="Verify your Football Coach account",
        intro_line="Your Football Coach verification code is",
        code=code,
        full_name=full_name,
    )


def send_password_reset_email(
    to_email: str,
    code: str,
    *,
    full_name: Optional[str] = None,
) -> None:
    _send_code_email(
        to_email,
        subject="Reset your Football Coach password",
        intro_line="Your Football Coach password reset code is",
        code=code,
        full_name=full_name,
    )
