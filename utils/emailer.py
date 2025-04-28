# utils/emailer.py

import smtplib
import os
import logging
from email.message import EmailMessage
import ssl
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def send_email(
    report_file_path: str,
    sender_email: str,
    sender_password: str,
    receiver_email: str,
    smtp_server: str,
    smtp_port: int
) -> Optional[str]:
    """
    Send an email with the report file as an attachment.

    Args:
        report_file_path (str): Path to the report file to attach.
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password or app password.
        receiver_email (str): Receiver's email address.
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP server port.

    Returns:
        Optional[str]: Error message if failed, None if successful.

    Raises:
        ValueError: If report file doesn't exist or is invalid.
        ConnectionError: If SMTP connection fails.
    """
    try:
        # Validate report file
        report_path = Path(report_file_path)
        if not report_path.exists():
            raise ValueError(f"Report file not found: {report_file_path}")
        if not report_path.is_file():
            raise ValueError(f"Invalid report file path: {report_file_path}")

        # Create email
        subject = "🚦 Traffic Sign Damage Report"
        body = "Please find the attached damage report for the detected traffic sign."

        em = EmailMessage()
        em['From'] = sender_email
        em['To'] = receiver_email
        em['Subject'] = subject
        em.set_content(body)

        # Attach the report
        try:
            with open(report_path, 'rb') as f:
                file_data = f.read()
                file_name = report_path.name
        except IOError as e:
            logger.error(f"Failed to read report file: {e}")
            raise ValueError(f"Failed to read report file: {e}")

        em.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

        # Secure the connection
        context = ssl.create_default_context()

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as smtp:
                smtp.starttls(context=context)
                smtp.login(sender_email, sender_password)
                smtp.send_message(em)
        except (smtplib.SMTPException, ConnectionError) as e:
            logger.error(f"SMTP error: {e}")
            raise ConnectionError(f"Failed to send email: {e}")

        logger.info(f"Email sent successfully with report: {file_name}")
        return None

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return str(e)
