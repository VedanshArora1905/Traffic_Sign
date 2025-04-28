# tests/test_emailer.py

import pytest
from pathlib import Path
from utils.emailer import send_email

def test_send_email_invalid_path():
    """Test sending email with invalid report path."""
    result = send_email(
        report_file_path="nonexistent.txt",
        sender_email="test@example.com",
        sender_password="password",
        receiver_email="receiver@example.com",
        smtp_server="smtp.example.com",
        smtp_port=587
    )
    assert result is not None
    assert "not found" in result.lower()

def test_send_email_invalid_credentials():
    """Test sending email with invalid credentials."""
    # Create a test report file
    test_report = Path("tests/data/test_report.txt")
    test_report.parent.mkdir(parents=True, exist_ok=True)
    test_report.write_text("Test report content")

    # Test with invalid credentials
    result = send_email(
        report_file_path=str(test_report),
        sender_email="invalid@example.com",
        sender_password="wrong_password",
        receiver_email="receiver@example.com",
        smtp_server="smtp.example.com",
        smtp_port=587
    )
    assert result is not None
    assert "failed" in result.lower()

    # Cleanup
    test_report.unlink()

def test_send_email_invalid_smtp():
    """Test sending email with invalid SMTP server."""
    # Create a test report file
    test_report = Path("tests/data/test_report.txt")
    test_report.parent.mkdir(parents=True, exist_ok=True)
    test_report.write_text("Test report content")

    # Test with invalid SMTP server
    result = send_email(
        report_file_path=str(test_report),
        sender_email="test@example.com",
        sender_password="password",
        receiver_email="receiver@example.com",
        smtp_server="invalid.smtp.server",
        smtp_port=587
    )
    assert result is not None
    assert "failed" in result.lower()

    # Cleanup
    test_report.unlink() 