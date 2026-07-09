import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email():
    sender_email = "your_email@gmail.com"
    receiver_email = "nikhilmundhra28@gmail.com"
    # Note: If using Gmail, you must use an App Password, not your regular account password.
    # See: https://support.google.com/accounts/answer/185833
    password = "your_app_password"

    subject = "Hello from Python!"
    body = "This is a simple SMTP email sent from a Python script."

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to Gmail's SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls() # Secure the connection
        server.login(sender_email, password)
        
        # Send the email
        server.send_message(msg)
        print("Email sent successfully!")
        
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()

if __name__ == "__main__":
    send_email()
