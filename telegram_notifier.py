import requests

TOKEN = '7805721792:AAGZ4WjU4yhzaC03cIK5c7goxACF_zn0OFk'
CHAT_ID = '7516642698'

def send_telegram_notification(message: str, image_path: str = None):
    """
    Sends a Telegram notification with a text message and optionally an image.
    
    :param message: The text message to send.
    :param image_path: Optional; the file path to the image to attach.
    :return: The JSON response from Telegram.
    """
    
    # Send the image if provided, with the message as a caption
    if image_path:
        send_photo_url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        with open(image_path, 'rb') as image_file:
            files = {'photo': image_file}
            data = {'chat_id': CHAT_ID, 'caption': message}
            response = requests.post(send_photo_url, data=data, files=files)
            result = response.json()
    # Send the text message if no image provided
    else:
        send_message_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        params = {'chat_id': CHAT_ID, 'text': message}
        response = requests.get(send_message_url, params=params)
        result = response.json()


    # return result