import smtplib
import os
from email.message import EmailMessage

class EmailAgent:
  """docstring for EmailAgent"""
  def __init__(self):
    self.smtp_server = os.getenv("SMTP_SERVER")
    self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
    self.from_email = os.getenv("FROM_EMAIL")

  def prepare_email(self, to_email: str, body: str):
    msg = EmailMessage()
    msg["From"] = self.from_email
    msg["To"] = "quality_control@example.com"
    msg["Subject"] = "Album Images Are Ready"
    msg.set_content(body)
    return msg

  def send_email(self, to_email: str, body: str):
    msg = self.prepare_email(to_email, body)
    with smtplib.SMTP(host=self.smtp_server, port=self.smtp_port) as server:
      server.send_message(msg)

  def preview_email(self, to_email: str, body: str) -> dict:
    """Preview email details before sending.

    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body text

    Returns:
        Dictionary with email details
    """
    return {
        'from': self.from_email,
        'to': to_email,
        'subject': "Album Images Are Ready",
        'body': body,
    }
