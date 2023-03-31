import logging
import pickle
import json
import re
import os.path
from time import time
from typing import Dict, List

import pandas as pd

from lib.ML import GenerateLabels
from lib.authorize import build_service
from lib.read_mails import read_n_mails

service = build_service()
logger = logging.getLogger(__name__)
EXC_INFO = True


def store_list_of_labels():
    """
    retrieve list of labels from user's Gmail and store them in a json file
    """
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        return

    label_dict = {label['id']: label['name'] for label in labels}
    with open('data/label_dict.json', 'w') as file:
        json.dump(label_dict, file)


def list_labels_from_old() -> Dict[str, str]:
    """
    list the labels from the data which was stored
    """
    if os.path.exists('data/label_dict.json'):
        with open('data/label_dict.json', 'r') as file:
            labels_dict = json.load(file)
    else:
        logger.error('File labels_dict not found')

    return labels_dict


default_labels_list = [
    key
    for key, value in list_labels_from_old().items()
    if key == value
]

label_name_list = [
    value for key, value in list_labels_from_old().items() if re.match('Label_[0-9]', key)
]


def create_labels():
    for label_name in label_name_list:
        try:
            logger.debug('creating new label')
            service.users().labels().create(
                userId='me', body={
                    "name": label_name
                }
            ).execute()
        except:
            logger.error('unable to create label', exc_info=EXC_INFO)

    logger.info('labels created successfully')


def set_label(msg_id: str, labels, removeLabels=False) -> bool:
    """
    will apply labels only on 1 mail
    :return: true if applied else false
    """
    if not labels:
        logger.debug('no label matches')
        return False

    logger.debug(f'applying label to msg id {msg_id}')

    if removeLabels:
        body = {
            "removeLabelIds": list(labels),
            "addLabelIds": []
        }

    else:
        body = {
            "removeLabelIds": [],
            "addLabelIds": list(labels)
        }
    try:
        service.users().messages().modify(userId='me', id=msg_id, body=body).execute()
        logger.debug('label successfully applied')
        return True
    except Exception as error:
        logger.debug('unable to apply label')
        logger.error(error)
        return False


def label_mails(mails_df: pd.DataFrame):
    """
    will set labels to all the messages in the given dataframe according to the model's predictions
    :param mails_df: dataframe of mails as it is received
    """
    label_generator = GenerateLabels()
    t0 = time()
    logger.info('model prediction started')
    label_names_list = label_generator.generate_labels(mails_df)
    label_ids_list = []

    for label_names in label_names_list:
        labels_dict = list_labels_from_old()
        label_ids = [
            list(labels_dict.keys())[list(labels_dict.values()).index(label_name)]
            for label_name in label_names
        ]
        label_ids_list.append(label_ids)

    logger.info(f'model generated labels in {time() - t0} seconds')
    ctr = 0

    for i, msg_id in enumerate(mails_df['id']):
        logger.debug(f'applying label to MailNo- {i}')
        status = set_label(msg_id, label_ids_list[i])
        if status:
            ctr += 1
    
    logger.debug(f'labeled {ctr} mails out of {len(mails_df)}')


def label_first_n_mails(n: int):
    """
    labels top n mails in the user's mailbox
    """
    mails_df = read_n_mails(n)
    label_mails(mails_df)


def reset_labels(mails_df: pd.DataFrame):
    labels_list: List[List[str]] = [
        [
            label
            for label in st.split(',')
            if label not in default_labels_list
        ]
        for st in mails_df['labels']
    ]
    # as default labels don't have to be removed

    for i, msg_id in enumerate(mails_df['id']):
        set_label(msg_id, labels_list[i], removeLabels=True)
