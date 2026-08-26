#!/usr/bin/env python3
"""Send the daily summary by email over SMTP.

SMTP, not IMAP: IMAP reads a mailbox, SMTP sends. Same Gmail app-password setup either
way, but only one of them delivers mail.

CREDENTIALS -- put these in .env, never in source:

    SMTP_ADDRESS       the sending account, e.g. romado33@gmail.com
    SMTP_APP_PASSWORD  a Google App Password, NOT the account password
    SMTP_HOST          optional, default smtp.gmail.com
    SMTP_PORT          optional, default 587 (STARTTLS)
    SUMMARY_TO         optional default recipient

A Google App Password requires 2-Step Verification on the account, and is generated at
myaccount.google.com/apppasswords. Gmail rejects the normal account password over SMTP,
so a plain password will fail authentication no matter how correct it looks.

WHAT THIS MEANS FOR THE DATA, said once and plainly: emailing a summary of work
conversations to a personal mailbox puts that content in personal custody permanently.
It is a larger step than the API call that produced it -- an API request is transient,
a mailbox is an archive. That is the operator's decision to make, but it should be a
decision rather than a side effect of wiring up a pipeline.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, formatdate

logger = logging.getLogger(__name__)

DEFAULT_HOST = "smtp.gmail.com"
DEFAULT_PORT = 587


def _first_env(*names: str) -> str:
    """First non-empty value among these environment variables."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def default_recipient() -> str:
    """Where summaries go. Falls back to the sending account, which is the common case:
    you are mailing yourself, so requiring SUMMARY_TO as well is needless ceremony."""
    return _first_env(
        "SUMMARY_TO", "SMTP_ADDRESS", "GMAIL_EMAIL_ADDRESS", "GMAIL_ADDRESS"
    )


class MailError(RuntimeError):
    """Sending failed. Raised, never swallowed: a summary nobody received is worse than
    an error, because the pipeline would look like it succeeded."""


@dataclass(frozen=True)
class MailConfig:
    address: str
    app_password: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    sender_name: str = "AutoWork"
    timeout_sec: int = 30

    @classmethod
    def from_env(cls) -> "MailConfig":
        # Both naming conventions are accepted. GMAIL_* is what a person naturally
        # writes when setting up Gmail; SMTP_* is what the protocol is actually called
        # and what a non-Gmail provider would use. Rejecting one of them because the
        # other was documented is a pointless failure.
        address = _first_env("SMTP_ADDRESS", "GMAIL_EMAIL_ADDRESS", "GMAIL_ADDRESS")
        password = _first_env(
            "SMTP_APP_PASSWORD", "GMAIL_APP_PASSWORD", "SMTP_PASSWORD"
        )

        # Google displays App Passwords as four space-separated groups
        # ("abcd efgh ijkl mnop"). Copying that verbatim gives a 19-character string
        # that Gmail rejects, with an authentication error that looks exactly like a
        # wrong password. Strip whitespace rather than make the user debug it.
        password = "".join(password.split())

        missing = [
            name
            for name, value in (
                ("SMTP_ADDRESS (or GMAIL_EMAIL_ADDRESS)", address),
                ("SMTP_APP_PASSWORD (or GMAIL_APP_PASSWORD)", password),
            )
            if not value
        ]
        if missing:
            raise MailError(
                f"missing mail credentials: {', '.join(missing)}. Add them to .env. "
                f"The password must be a Google App Password "
                f"(myaccount.google.com/apppasswords), not the account password."
            )

        if len(password) != 16:
            logger.warning(
                "app password is %d characters after stripping spaces; Google App "
                "Passwords are 16. Authentication may fail.", len(password),
            )

        try:
            port = int(os.environ.get("SMTP_PORT", DEFAULT_PORT))
        except ValueError as exc:
            raise MailError(
                f"SMTP_PORT is not a number: {os.environ.get('SMTP_PORT')!r}"
            ) from exc

        return cls(
            address=address,
            app_password=password,
            host=os.environ.get("SMTP_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
            port=port,
        )


def build_message(
    *,
    sender: str,
    sender_name: str,
    recipient: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = formataddr((sender_name, sender))
    message["To"] = recipient
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    message.set_content(text_body)
    if html_body:
        message.add_alternative(html_body, subtype="html")
    return message


def send(
    recipient: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
    config: MailConfig | None = None,
) -> str:
    """Send one message. Returns a short confirmation string.

    Distinguishes the failure modes that matter, because "it didn't send" is not
    actionable: bad credentials, a rejected recipient, and an unreachable server need
    different fixes and the SMTP library reports them as different exceptions.
    """
    config = config or MailConfig.from_env()
    if not recipient.strip():
        raise MailError("no recipient given")

    message = build_message(
        sender=config.address,
        sender_name=config.sender_name,
        recipient=recipient,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )

    context = ssl.create_default_context()
    try:
        if config.port == 465:
            server: smtplib.SMTP = smtplib.SMTP_SSL(
                config.host, config.port, timeout=config.timeout_sec, context=context
            )
        else:
            server = smtplib.SMTP(config.host, config.port, timeout=config.timeout_sec)
        with server:
            if config.port != 465:
                server.starttls(context=context)
            server.login(config.address, config.app_password)
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise MailError(
            f"SMTP rejected the credentials for {config.address}. With Gmail this "
            f"almost always means the value in SMTP_APP_PASSWORD is the account "
            f"password rather than an App Password, or 2-Step Verification is not "
            f"enabled on the account. Server said: {exc.smtp_error!r}"
        ) from exc
    except smtplib.SMTPRecipientsRefused as exc:
        raise MailError(f"recipient refused: {exc.recipients}") from exc
    except smtplib.SMTPException as exc:
        raise MailError(f"SMTP error talking to {config.host}:{config.port}: {exc}") from exc
    except OSError as exc:
        raise MailError(
            f"cannot reach {config.host}:{config.port}: {exc}. "
            f"Corporate networks often block outbound SMTP."
        ) from exc

    # Never log the password, and never log the body -- it is the sensitive part.
    logger.info("sent %r to %s via %s", subject, recipient, config.host)
    return f"sent to {recipient} via {config.host}:{config.port}"
