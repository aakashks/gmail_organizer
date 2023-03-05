import logging
import pickle
from time import time
from typing import Dict

import pandas as pd

from src.lib.ML import GenerateLabels
from src.lib.authorize import build_service
from src.lib.read_mails import read_n_mails

service = build_service()
logger = logging.getLogger(__name__)


def store_list_of_labels():
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        return

    label_dict = {label['id']: label['name'] for label in labels}
    with open('../../data/label_dict.pickle', 'wb') as file:
        pickle.dump(label_dict, file)


def list_labels_from_old() -> Dict[str, str]:
    with open('../../data/label_dict.pickle', 'rb') as file:
        labels_dict = pickle.load(file)
    return labels_dict


def create_labels(): ...


def set_label(msg_id: str, labels: tuple):
    """
    will apply labels only on 1 mail
    """
    if not labels:
        logger.debug('no label matches')
        return

    logger.debug(f'applying label to msg id {msg_id}')
    body = {
        "removeLabelIds": [],
        "addLabelIds": list(labels)
    }
    service.users().messages().modify(userId='me', id=msg_id, body=body).execute()
    logger.info('label successfully applied')


def label_mails(mails_df: pd.DataFrame):
    """
    will set labels to all the messages in the given dataframe according to the model's predictions
    :param mails_df: dataframe of mails as it is received
    """
    knn_label_generator = GenerateLabels()
    t0 = time()
    logger.info('model prediction started')
    labels_list = knn_label_generator.generate_labels(mails_df)
    logger.info(f'model generated labels in {time()-t0} seconds')
    for i, msg_id in enumerate(mails_df['id']):
        set_label(msg_id, labels_list[i])


def label_first_n_mails(n: int):
    """
    labels top n mails in the user's mailbox
    """
    mails_df = read_n_mails(n)
    label_mails(mails_df)
