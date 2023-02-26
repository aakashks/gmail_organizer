import base64
import logging
import re
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup

from src.lib.authorize import build_service

logger = logging.getLogger(__name__)

# Call the Gmail API
service = build_service()
TOTAL_MAILS = 981


def list_labels():
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        return []

    label_list = [label['name'] for label in labels]
    return label_list


all_labels = list_labels()


def read_n_mails(no_of_mails=1, max_mail_once=100):
    """
    give dataframe containing specified no of mails
    :param max_mail_once: int
    :param no_of_mails: int
    :return: pd.DataFrame
    """
    mail_ctr = 0
    read_mails = 0

    # at maximum only 500 mails can be read once

    if no_of_mails <= max_mail_once:
        result = service.users().messages().list(maxResults=no_of_mails, userId='me').execute()
        # messages is a list of dictionaries where each dictionary contains a message id.
        messages = result.get('messages')

    # if mails are more than 500, next page token is required for reading next page
    else:
        result = service.users().messages().list(maxResults=max_mail_once, userId='me').execute()
        # messages is a list of dictionaries where each dictionary contains a message id.
        messages = result.get('messages')
        next_page_token = result.get('nextPageToken')
        n = no_of_mails - max_mail_once
        while n > 0:
            logger.debug('using page tokens!')
            no_of_results = max_mail_once if n >= max_mail_once else n
            next_result = service.users().messages().list(
                maxResults=no_of_results, userId='me', pageToken=next_page_token).execute()

            messages.append(next_result.get('messages'))
            if n // max_mail_once:
                next_page_token = next_result.get('nextPageToken')

            n //= max_mail_once

    # dictionary to store mails
    messages_dict = defaultdict(list)
    # labels_dict = {label: [] for label in all_labels}

    logger.debug(f'listed {len(messages)} messages')

    # iterate through all the messages
    for msg in messages:
        mail_ctr += 1
        # Use try-except to avoid any Errors
        try:
            # Get the message from its id
            txt = service.users().messages().get(userId='me', id=msg['id']).execute()

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
                    subject = format_text(d['value'])
                if d['name'] == 'From':
                    sender = format_address(d['value'])
                if d['name'] == 'To':
                    receiver = format_address(d['value'])

            # The Body of the message is in Encrypted format. So, we have to decode it.
            # Get the data and decode it with base 64 decoder.
            if payload.get('parts'):
                parts = payload.get('parts')[0]
                data = parts['body']['data']
            elif payload.get('body'):
                data = payload.get('body')['data']

            data = data.replace("-", "+").replace("_", "/")
            # decrypting the message body
            decoded_data = base64.b64decode(data)

            # Now, the data obtained is in lxml. So, we will parse
            # it with BeautifulSoup library
            soup = BeautifulSoup(decoded_data, "lxml")
            body = str(soup.body())[4:-5]
            formatted_body = format_text(body)

            # adding data in the dictionary
            messages_dict['id'].append(msg['id'])
            # ignoring date as its format is unfit and its irrelevant
            # messages_dict['date'].append(date)
            messages_dict['sender'].append(sender)
            messages_dict['receiver'].append(receiver)
            messages_dict['subject'].append(subject)
            messages_dict['body'].append(formatted_body)

            # # filling up labels as a numerical data
            # for label in all_labels:
            #     if label in labels:
            #         labels_dict[label].append('1')
            #     else:
            #         labels_dict[label].append('0')

            # filling labels into one col as categorical data
            label_list_str = ','.join(list(labels))
            messages_dict['labels'].append(label_list_str)

            logger.debug('read mail successfully')
            logger.debug(f'read MailNo- {mail_ctr}')
            read_mails += 1
        except Exception as error:
            logger.error(f'{error}')
            logger.debug(f'unable to read MailNo- {mail_ctr}')

        logger.info(f'read {read_mails} out of {no_of_mails} mails')

    # dataframe to store the messages
    messages_df = pd.DataFrame(messages_dict)
    # labels_df = pd.DataFrame(labels_dict)
    # mails_df = pd.concat([messages_df, labels_df], axis=1)
    return messages_df


def store_all_mails(test_mode=False):
    if test_mode:
        logger.info('reading some mails for testing')
        lst_dfs = [read_n_mails(21, 5)]

    else:
        logger.info('reading all mails stored')
        lst_dfs = [read_n_mails(TOTAL_MAILS)]

    all_mails_df = pd.concat(lst_dfs)
    all_mails_df.to_csv('../../temp/test1.csv', sep='~')


def format_text(text):
    """
    :param text: str
    :return: str
    will return a simplified format of the body with :: separated words
    """
    regex = re.compile('\s+')
    list_of_words = regex.split(text)
    formatted_text = ' '.join(list_of_words)
    return formatted_text


def format_address(text):
    """
    :param text: str
    :return: str
    extract the email addresses from sender/receiver details
    """
    regex = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}', flags=re.IGNORECASE)
    address_list = regex.findall(text)
    address_list_str = ','.join(address_list)
    return address_list_str
