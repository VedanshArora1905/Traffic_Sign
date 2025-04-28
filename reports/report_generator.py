import os
from datetime import datetime

def generate_report(bbox, original_label, severity, damage_type, output_dir):
    """
    Generates a simple text report summarizing the detected damage.

    Args:
        bbox (tuple): Bounding box of the detected sign.
        original_label (str): Predicted label for the sign class.
        severity (str): Severity of the damage.
        damage_type (str): Type of damage.
        output_dir (str): Directory to save the report.

    Returns:
        report_path (str): Path to the generated report file.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Report content
    report_content = f"""
🚦 Traffic Sign Damage Report
------------------------------
Timestamp         : {now}
Detected Sign     : {original_label}
Bounding Box      : {bbox}
Damage Severity   : {severity}
Damage Type       : {damage_type}

Recommended Action: {suggest_action(severity)}

Thank you.
    """

    # Save report
    os.makedirs(output_dir, exist_ok=True)

    report_filename = f"report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, 'w') as f:
        f.write(report_content.strip())

    print(f"✅ Report generated at {report_path}")
    return report_path

def suggest_action(severity):
    """
    Suggest action based on severity.
    """
    if severity == "Low":
        return "Repaint or Clean Minor Damages."
    elif severity == "Medium":
        return "Consider Partial Replacement or Repair."
    elif severity == "High":
        return "Immediate Full Replacement Recommended."
    else:
        return "Assessment Required."