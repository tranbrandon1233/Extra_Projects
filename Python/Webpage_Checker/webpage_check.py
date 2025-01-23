import requests
import os
import time
import difflib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import dotenv_values

config = dotenv_values(".env")

# Email configuration
sender_email = config['SENDER_EMAIL']
receiver_email = config['RECEIVER_EMAIL']
password = config['SENDER_PASSWORD']
smtp_server = "smtp.gmail.com"
port = 465  # For SSL

# URL of the webpage to monitor
url = "https://reddit.com/"

# Path to the file where the last content is saved
saved_content_file = "saved_page.html"

def fetch_page(url):
    """
    Fetches the content of the webpage at the given URL.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    return response.text

def send_email(diff_text):
    """
    Sends an email containing the diff_text to the receiver_email.
    """
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Webpage Content Changed"

    # Attach the diff text to the email message
    message.attach(MIMEText(diff_text, "plain"))

    # Send the email via SMTP server
    with smtplib.SMTP_SSL(smtp_server, port) as server:
        server.login(sender_email, password)
        server.send_message(message)
        print("Email sent successfully!")

def save_content(content, file_path):
    """
    Saves the content to a file specified by file_path.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def load_content(file_path):
    """
    Loads and returns the content from the file specified by file_path.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

def compare_contents(old_content, new_content):
    """
    Compares old_content and new_content and returns the differences.
    """
    diff = difflib.unified_diff(
        old_content.splitlines(),
        new_content.splitlines(),
        fromfile='Previous Content',
        tofile='Current Content',
        lineterm=''
    )
    return '\n'.join(diff)

def monitor_webpage():
    """
    Monitors the webpage for changes every 10 seconds.
    """
    while True:
        try:
            print("Checking for changes...")
            # Fetch the current content of the webpage
            current_content = fetch_page(url)
            
            # Load the previously saved content
            previous_content = load_content(saved_content_file)
            
            # If there is previous content, compare it with the current content
            if previous_content is not None:
                if current_content != previous_content:
                    print("Change detected! Preparing to send email...")
                    # Get the differences between the previous and current content
                    diff_text = compare_contents(previous_content, current_content)
                    
                    # Send an email with the differences
                    send_email(diff_text)
                else:
                    print("No changes detected.")
            else:
                print("No previous content found. Saving current content for future comparisons.")
            
            # Save the current content for the next comparison
            save_content(current_content, saved_content_file)
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        # Wait for 10 seconds before checking again
        time.sleep(10)

if __name__ == "__main__":
    monitor_webpage()