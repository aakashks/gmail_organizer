from authorize import build_service
import base64
from bs4 import BeautifulSoup

# Call the Gmail API
service = build_service()


def read_n_mails(n=1):
    result = service.users().messages().list(maxResults=n, userId='me').execute()
    messages = result.get('messages')
    # messages is a list of dictionaries where each dictionary contains a message id.

    # iterate through all the messages
    for msg in messages:
        # Get the message from its id
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()

        # Use try-except to avoid any Errors
        try:
            # Get value of 'payload' from dictionary 'txt'
            payload = txt['payload']
            headers = payload['headers']
            # get list of labels of the email
            labels = txt['labelIds']

            # Look for metadata of Email in the headers
            # which is a list of dictionaries
            for d in headers:
                if d['name'] == 'Date':
                    date = d['value']
                if d['name'] == 'Subject':
                    subject = d['value']
                if d['name'] == 'From':
                    sender = d['value']
                if d['name'] == 'To':
                    receiver = d['value']

            # The Body of the message is in Encrypted format. So, we have to decode it.
            # Get the data and decode it with base 64 decoder.
            parts = payload.get('parts')[0]
            data = parts['body']['data']
            data = data.replace("-", "+").replace("_", "/")
            # decrypting the message body
            decoded_data = base64.b64decode(data)

            # Now, the data obtained is in lxml. So, we will parse
            # it with BeautifulSoup library
            soup = BeautifulSoup(decoded_data, "lxml")
            body = soup.body()

            # Printing the subject, sender's email and message
            print("Subject: ", subject)
            print("From: ", sender)
            print("Message: ", body)

        except:
            print('error')
