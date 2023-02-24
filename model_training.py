# will fetch all mails from my mailbox to train the model

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from tqdm import tqdm
import pandas as pd

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',  # for modifying labels of messages
    'https://www.googleapis.com/auth/gmail.labels'  # for managing labels
]

no_of_messages = 800


def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
        page_token = ''
        for i in tqdm(range(no_of_messages // 100 + 1), desc='loading messages'):
            results = service.users().messages().list(userId='me').execute(
                pageToken=page_token
            )
            message_id_list = [message.id
                               for message in results.get('messages', [])
                               ]
            page_token = results.get('nextPageToken')

            raw_message_df = pd.DataFrame()
            for message_id in message_id_list:
                message_result = service.users().messages().get(
                    userId='me', id=message_id
                ).execute(format='full')
                message = pd.DataFrame({message_id: message_result})
                raw_message_df.append(message)

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    main()
