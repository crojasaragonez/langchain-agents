import smtplib
from email.message import EmailMessage

# Create the email
msg = EmailMessage()
msg["From"] = "sender@example.com"
msg["To"] = "recipient@example.com"
msg["Subject"] = "Hello World"
msg.set_content("Hello World")

# Send the email via local SMTP server
with smtplib.SMTP(host="localhost", port=1025) as server:
    server.send_message(msg)

print("Email sent successfully!")
