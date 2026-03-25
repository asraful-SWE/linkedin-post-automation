"""
Email Service - Sends approval emails for generated posts
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


logger = logging.getLogger(__name__)


class EmailService:
    """SMTP email sender for approval workflow"""

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

        self.email_from = os.getenv("EMAIL_FROM", self.smtp_user or "noreply@example.com")
        self.email_to = os.getenv("APPROVAL_EMAIL_TO", "")
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

    def is_configured(self) -> bool:
        return all([self.smtp_host, self.smtp_port, self.email_to])

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

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as server:
                if self.smtp_use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.email_from, [self.email_to], msg.as_string())
            logger.info(f"Approval email sent for post_id={post_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send approval email for post_id={post_id}: {e}")
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