import requests
import os

# load api keys from file
KEYS_PATH = os.path.expanduser('~/telegram_api_keys/telegram_keys')
with open(KEYS_PATH, 'r') as keys_file:
    keys = {}
    for line in keys_file:
        line = line.strip()
        if line and '=' in line:
            key, value = line.split('=', 1)
            keys[key.strip()] = value.strip().strip("'").strip('"')
    TOKEN = keys.get('TOKEN')
    CHAT_ID = keys.get('CHAT_ID')
    
def send_telegram_notification(message: str, image_path: str = None):
    """
    Sends a Telegram notification with a text message and optionally an image.
    
    :param message: The text message to send.
    :param image_path: Optional; the file path to the image to attach.
    """
    
    # Send the image if provided, with the message as a caption
    if image_path:
        send_photo_url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        with open(image_path, 'rb') as image_file:
            files = {'photo': image_file}
            data = {'chat_id': CHAT_ID, 'caption': message}
            response = requests.post(send_photo_url, data=data, files=files)
    # Send the text message if no image provided
    else:
        send_message_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        params = {'chat_id': CHAT_ID, 'text': message}
        response = requests.get(send_message_url, params=params)