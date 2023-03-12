import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import logging

logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',  # for modifying labels of messages
    'https://www.googleapis.com/auth/gmail.labels'  # for managing labels
]


def build_service():
    """
    authorize user using gmail api and create service
    :return:
    """
    if not os.path.exists('conf/credentials.json'):
        raise Exception('credentials.json file not created!')

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('conf/token.json'):
        creds = Credentials.from_authorized_user_file('conf/token.json', SCOPES)
        logger.info('Authorized user')

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        logger.info('Authorize yourself')
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'conf/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('conf/token.json', 'w') as token:
            token.write(creds.to_json())

        logger.info('Successfully Authorized !')

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
        return service

    except HttpError as error:
        logger.error('Authorization error or service build error')
        logger.error(f'An error occurred: {error}')
