import subprocess
import os
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def send_email_notification(failed_scripts):
    sender = "nsteenholdt@gmail.com"
    recipient = "nsteenholdt@gmail.com"
    subject = (
        "All scripts completed successfully"
        if not failed_scripts
        else f"⚠️ Some scripts failed: {', '.join(failed_scripts)}"
    )
    body = (
        "All scripts finished without errors. "
        if not failed_scripts
        else f"The following scripts encountered errors:\n\n" + "\n".join(failed_scripts)
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, "jlaa omrj sjoq sifu")  # Use app password here
            server.sendmail(sender, [recipient], msg.as_string())
        print("Email sent!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def run_script(script_path):
    ext = os.path.splitext(script_path)[1]
    if ext == ".py":
        return subprocess.run(["python", script_path], capture_output=True, text=True)
    elif ext == ".ipynb":
        return subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                script_path,
            ],
            capture_output=True,
            text=True
        )
    else:
        raise ValueError(f"Unsupported script type: {script_path}")

# --- Scripts to run ---
scripts = [
    "count_gender_word_new.py",
    "gendered_titles_new.py",
    "gender_count.ipynb"
]

# --- Working directory ---
working_dir = os.path.dirname(os.path.abspath(__file__))

# --- Start total timer ---
total_start = time.time()
print(f"Starting script run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
failed_scripts = []

# --- Main execution ---
for script in scripts:
    script_path = os.path.join(working_dir, script)
    print(f"\n Running: {script} at {datetime.now().strftime('%H:%M:%S')}")
    start = time.time()

    try:
        result = run_script(script_path)
        elapsed = time.time() - start

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        print(f"Finished: {script} in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error in {script}: {e}")
        failed_scripts.append(f"{script} - {str(e)}")

# --- Send email notification ---
send_email_notification(failed_scripts)

# --- Wrap up ---
total_elapsed = time.time() - total_start
print(f"\nTotal run time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
print("Sending Mac to sleep...")
os.system("pmset sleepnow")
#time.sleep(10)
