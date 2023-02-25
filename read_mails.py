from authorize import build_service
import base64
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict

# Call the Gmail API
service = build_service()
TOTAL_MAILS = 700

def list_labels():
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        return []

    label_list = [label['name'] for label in labels]
    return label_list


all_labels = list_labels()


def read_n_mails(n=1):
    """
    give dataframe containing specified no of mails
    :param n: int  #maximum 200
    :return: pd.DataFrame
    """

    result = service.users().messages().list(maxResults=n, userId='me').execute()
    # messages is a list of dictionaries where each dictionary contains a message id.
    messages = result.get('messages')
    # dictionary to store mails
    messages_dict = defaultdict(list)
    labels_dict = {label: [] for label in all_labels}

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
            body = str(soup.body())[4:-5]
            formatted_body = format_body(body)

            # adding data in the dictionary
            messages_dict['id'].append(msg['id'])
            messages_dict['date'].append(date)
            messages_dict['sender'].append(sender)
            messages_dict['receiver'].append(receiver)
            messages_dict['subject'].append(subject)
            messages_dict['body'].append(formatted_body)

            # filling up labels as a numerical data
            for label in labels:
                labels_dict[label].append(1)

        except:
            pass

    # dataframe to store the messages
    messages_df = pd.DataFrame(messages_dict)
    labels_df = pd.DataFrame(labels_dict)
    mails_df = pd.concat([messages_df, labels_df])
    return mails_df

def format_body(body):
    return body

def write_all_mails_to_hdf():
    lst_dfs = []
    for mail_count in range(TOTAL_MAILS//200):
        lst_dfs.append(read_n_mails(200))

    all_mails_df = pd.concat(lst_dfs)
    all_mails_df.to_hdf('./data/training_data.hdf')