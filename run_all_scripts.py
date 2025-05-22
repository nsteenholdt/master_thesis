import subprocess
import os
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def send_email_notification(success=True):
    sender = "nsteenholdt@gmail.com"
    recipient = "nsteenholdt@gmail.com"
    subject = "✅ Script completed successfully" if success else "❌ Script failed"
    body = "Your script has finished running."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, "jlaa omrj sjoq sifu")
            server.sendmail(sender, [recipient], msg.as_string())
        print("📧 Email sent!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# --- Scripts to run ---
scripts = [
    "extract_json_to_csv.py",
    "filtering_lang_descriptions.py",
    "other_gendering_script.py",
    "counting_gendered_words.py",
    "count_chat_words.py",
    "gendered_titles_over_time.py",
    "job_title_gender_bias_analysis.py"
]

# --- Working directory ---
working_dir = os.path.dirname(os.path.abspath(__file__))

# --- Start total timer ---
total_start = time.time()
print(f"🚀 Starting script run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Main execution ---
try:
    for script in scripts:
        script_path = os.path.join(working_dir, script)
        print(f"\n▶️ Running: {script} at {datetime.now().strftime('%H:%M:%S')}")
        start = time.time()

        subprocess.run(["python", script_path], check=True)

        elapsed = time.time() - start
        print(f"✅ Finished: {script} in {elapsed:.2f} seconds")

    send_email_notification(success=True)

except Exception as e:
    print(f"❌ Script run failed: {e}")
    send_email_notification(success=False)

finally:
    total_elapsed = time.time() - total_start
    print(f"\n⏱️ Total run time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    
    print("😴 Sending Mac to sleep...")
    os.system("pmset sleepnow")  # macOS-specific sleep command

    time.sleep(10)  # Optional: give the system a moment to finish logging
