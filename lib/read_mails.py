import base64
import logging
import re
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup

from lib.authorize import build_service

logger = logging.getLogger(__name__)

# Call the Gmail API
service = build_service()
TOTAL_MAILS = 993
MAX_MAIL_LIST = 500
EXC_INFO = True


def read_n_mails(no_of_mails: int) -> pd.DataFrame:
    """
    gives a dataframe containing specified no of mails
    :param no_of_mails: int
    :return: pd.DataFrame
    """
    mail_ctr = -1
    read_mails = 0

    # at maximum only 500 mails can be read once

    if no_of_mails <= MAX_MAIL_LIST:
        result = service.users().messages().list(maxResults=no_of_mails, userId='me').execute()
        # messages is a list of dictionaries where each dictionary contains a message id.
        messages = result.get('messages')

    # if mails are more than 500, next page token is required for reading next page
    else:
        result = service.users().messages().list(maxResults=MAX_MAIL_LIST, userId='me').execute()
        # messages is a list of dictionaries where each dictionary contains a message id.
        messages = result.get('messages')
        n = no_of_mails - MAX_MAIL_LIST
        while n > 0:
            next_page_token = result.get('nextPageToken')
            logger.debug('using page tokens!')
            no_of_results = MAX_MAIL_LIST if n >= MAX_MAIL_LIST else n
            result = service.users().messages().list(
                maxResults=no_of_results, userId='me', pageToken=next_page_token).execute()

            messages.extend(result.get('messages'))
            n = n - MAX_MAIL_LIST

    # dictionary to store mails
    messages_dict = defaultdict(list)
    # labels_dict = {label: [] for label in all_labels}

    logger.debug(f'listed {len(messages)} messages')

    # iterate through all the messages
    for msg in messages:
        mail_ctr += 1
        # Use try-except to avoid any Errors
        try:
            _read_mail(msg, messages_dict)
            logger.debug(f'read MailNo- {mail_ctr}')
            read_mails += 1
        except:
            logger.error('problem reading the mail', exc_info=EXC_INFO)
            logger.debug(f'----------unable to read MailNo- {mail_ctr}--------------!!!!!')

        logger.info(f'read {read_mails} out of {no_of_mails} mails')

    # dataframe to store the messages
    messages_df = pd.DataFrame(messages_dict)
    # labels_df = pd.DataFrame(labels_dict)
    # mails_df = pd.concat([messages_df, labels_df], axis=1)
    return messages_df


def _read_mail(msg, messages_dict):
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
        parts0 = payload.get('parts')[0]
        if parts0.get('body').get('data'):
            data = parts0['body']['data']
        elif parts0.get('parts')[1].get('body').get('data'):
            data = parts0.get('parts')[1].get('body').get('data')
        elif parts0.get('parts')[1].get('parts')[0].get('body').get('data'):
            data = parts0.get('parts')[1].get('parts')[0].get('body').get('data')

    elif payload.get('body'):
        data = payload.get('body')['data']

    # decrypting the message body
    decoded_data = base64.b64decode(data, '-_')

    # Now, the data obtained is in lxml. So, we will parse
    # it with BeautifulSoup library
    soup = BeautifulSoup(decoded_data, 'lxml')
    body = soup.text
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


def store_n_mails(no_of_mails=TOTAL_MAILS):
    """
    stores all mails into 1 csv at once
    :param no_of_mails: total mails to be read
    :return: 
    """
    logger.info('reading all mails stored')
    df = read_n_mails(no_of_mails)
    df.to_csv('data/training_data.csv', sep='~')


def format_text(text):
    """
    will return a simplified format of the body with space separated words
    """
    # remove tilda as it is the delimiter and any extra whitespace
    return re.sub(r'[\s~]+', ' ', text).strip()


def format_address(text):
    """
    extract the email addresses from sender/receiver details
    """
    regex = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}', flags=re.IGNORECASE)
    address_list = regex.findall(text)
    address_list_str = ','.join(address_list)
    return address_list_str
