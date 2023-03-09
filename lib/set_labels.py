import logging
import pickle
import json
from time import time
from typing import Dict, List

import pandas as pd

from lib.ML import GenerateLabels
from lib.authorize import build_service
from lib.read_mails import read_n_mails

service = build_service()
logger = logging.getLogger(__name__)


def store_list_of_labels():
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        return

    label_dict = {label['id']: label['name'] for label in labels}
    with open('data/label_dict.pickle', 'wb') as file:
        pickle.dump(label_dict, file)


def list_labels_from_old() -> Dict[str, str]:
    with open('data/label_dict.txt', 'r') as file:
        label_str = file.read()

    labels_dict = json.loads(label_str.replace('\'', '\"'))
    return labels_dict


default_labels_list = [
    key
    for key, value in list_labels_from_old().items()
    if key == value
]


def create_labels(): ...


def set_label(msg_id: str, labels, removeLabels=False):
    """
    will apply labels only on 1 mail
    """
    if not labels:
        logger.debug('no label matches')
        return

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
    except Exception as error:
        logger.debug('unable to remove label')
        logger.error(error)


def label_mails(mails_df: pd.DataFrame):
    """
    will set labels to all the messages in the given dataframe according to the model's predictions
    :param mails_df: dataframe of mails as it is received
    """
    knn_label_generator = GenerateLabels()
    t0 = time()
    logger.info('model prediction started')
    labels_list = knn_label_generator.generate_labels(mails_df)
    logger.info(f'model generated labels in {time() - t0} seconds')
    for i, msg_id in enumerate(mails_df['id']):
        set_label(msg_id, labels_list[i])


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
