"""
Email Service - Sends approval emails for generated posts
"""

import logging
import os
import smtplib
from typing import List, Tuple
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests


logger = logging.getLogger(__name__)


class EmailService:
    """SMTP email sender for approval workflow"""

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        # Support both new SMTP_* keys and legacy EMAIL_* keys.
        self.smtp_user = os.getenv("SMTP_USER", "") or os.getenv("EMAIL_ADDRESS", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "") or os.getenv("EMAIL_PASSWORD", "")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        self.smtp_use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() == "true"

        # Optional HTTPS fallback (works when SMTP ports are blocked).
        self.resend_api_key = os.getenv("RESEND_API_KEY", "")
        self.resend_from = os.getenv("RESEND_FROM", "")

        self.email_from = os.getenv("EMAIL_FROM", self.smtp_user or "noreply@example.com")
        self.email_to = os.getenv("APPROVAL_EMAIL_TO", "")
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

    def is_configured(self) -> bool:
        smtp_ready = bool(self.smtp_host and self.smtp_user and self.smtp_password)
        resend_ready = bool(self.resend_api_key)
        return bool(self.email_to and (smtp_ready or resend_ready))

    def send_post_approval_email(self, post_id: int, topic: str, content: str, token: str) -> bool:
        """Send approval email containing approve/reject/action links"""
        if not self.is_configured():
            logger.warning("Email service is not configured. Skipping approval email send.")
            return False

        approve_url = f"{self.base_url}/approve-post/{post_id}?token={token}"
        reject_url = f"{self.base_url}/reject-post/{post_id}?token={token}"
        form_url = f"{self.base_url}/approval-form/{post_id}?token={token}"

        html = self._build_approval_email_html(
            topic=topic,
            content=content,
            approve_url=approve_url,
            reject_url=reject_url,
            form_url=form_url,
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Post Approval Required: {topic}"
        msg["From"] = self.email_from
        msg["To"] = self.email_to
        msg.attach(MIMEText(html, "html", "utf-8"))

        # Prefer Resend when configured, because it uses HTTPS (port 443).
        if self.resend_api_key:
            if self._send_via_resend(msg):
                logger.info(f"Approval email sent for post_id={post_id} via Resend")
                return True
            logger.warning("Resend send failed, falling back to SMTP")

        if self._send_via_smtp(msg):
            logger.info(f"Approval email sent for post_id={post_id} via SMTP")
            return True

        logger.error(
            "Failed to send approval email for post_id=%s. "
            "Configure working SMTP credentials or RESEND_API_KEY.",
            post_id,
        )
        return False

    def _smtp_attempts(self) -> List[Tuple[str, int, bool, bool]]:
        """
        Build SMTP fallback attempts as tuples:
        (host, port, use_ssl, use_tls)
        """
        attempts: List[Tuple[str, int, bool, bool]] = []
        if not self.smtp_host:
            return attempts

        attempts.append((self.smtp_host, self.smtp_port, self.smtp_use_ssl, self.smtp_use_tls))

        # Common fallback: if 587 fails, try 465 SSL; if 465 fails, try 587 TLS.
        if self.smtp_port == 587:
            attempts.append((self.smtp_host, 465, True, False))
        elif self.smtp_port == 465:
            attempts.append((self.smtp_host, 587, False, True))

        return attempts

    def _send_via_smtp(self, msg: MIMEMultipart) -> bool:
        if not (self.smtp_host and self.smtp_user and self.smtp_password):
            logger.warning("SMTP not fully configured. Missing host/user/password.")
            return False

        for host, port, use_ssl, use_tls in self._smtp_attempts():
            try:
                if use_ssl:
                    server = smtplib.SMTP_SSL(host, port, timeout=20)
                else:
                    server = smtplib.SMTP(host, port, timeout=20)

                with server:
                    server.ehlo()
                    if use_tls and not use_ssl:
                        server.starttls()
                        server.ehlo()
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.email_from, [self.email_to], msg.as_string())
                return True
            except Exception as e:
                logger.warning(
                    "SMTP send attempt failed host=%s port=%s ssl=%s tls=%s error=%s",
                    host,
                    port,
                    use_ssl,
                    use_tls,
                    e,
                )

        return False

    def _send_via_resend(self, msg: MIMEMultipart) -> bool:
        if not self.resend_api_key:
            return False

        from_email = self.resend_from or self.email_from
        html_body = ""
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                html_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                break

        payload = {
            "from": from_email,
            "to": [self.email_to],
            "subject": msg["Subject"],
            "html": html_body,
        }
        headers = {
            "Authorization": f"Bearer {self.resend_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                "https://api.resend.com/emails",
                json=payload,
                headers=headers,
                timeout=20,
            )
            if 200 <= response.status_code < 300:
                return True
            logger.warning("Resend send failed: %s - %s", response.status_code, response.text[:300])
            return False
        except Exception as e:
            logger.warning("Resend send exception: %s", e)
            return False

    def _build_approval_email_html(
        self,
        topic: str,
        content: str,
        approve_url: str,
        reject_url: str,
        form_url: str,
    ) -> str:
        safe_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"""
<!doctype html>
<html>
  <body style=\"font-family: Arial, Helvetica, sans-serif; background: #f7f7f7; padding: 24px;\">
    <div style=\"max-width: 760px; margin: 0 auto; background: #fff; border-radius: 12px; border: 1px solid #e9e9e9; padding: 24px;\">
      <h2 style=\"margin-top: 0;\">LinkedIn Post Approval Required</h2>
      <p><strong>Topic:</strong> {topic}</p>

      <div style=\"margin: 16px 0; padding: 16px; border: 1px solid #ececec; border-radius: 8px; background: #fafafa; white-space: pre-wrap;\">{safe_content}</div>

      <div style=\"margin: 20px 0;\">
        <a href=\"{approve_url}\" style=\"display:inline-block;padding:12px 20px;background:#14833b;color:#fff;text-decoration:none;border-radius:8px;margin-right:12px;\">Approve Post</a>
        <a href=\"{reject_url}\" style=\"display:inline-block;padding:12px 20px;background:#b91c1c;color:#fff;text-decoration:none;border-radius:8px;\">Reject Post</a>
      </div>

      <div style=\"margin-top: 20px; padding: 14px; border: 1px dashed #cfcfcf; border-radius: 8px;\">
        <p style=\"margin: 0 0 8px 0;\"><strong>Add Image (optional)</strong></p>
        <p style=\"margin: 0 0 10px 0;\">You can add an image URL or upload an image before approval:</p>
        <a href=\"{form_url}\" style=\"display:inline-block;padding:10px 14px;background:#0e7490;color:#fff;text-decoration:none;border-radius:8px;\">Open Approval Form</a>
      </div>
    </div>
  </body>
</html>
"""