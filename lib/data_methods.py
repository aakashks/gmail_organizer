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
from lib.set_labels import list_labels_from_old

logger = logging.getLogger(__name__)
EXC_INFO = False


def write_label_names(label_id_series: pd.Series, exc_list=[]) -> pd.Series:
    """
    :param exc_list: a list of label_ids among the default ones which are required
    will convert comma separated label_ids into label_names for a series of strings
    """
    labels_dict = list_labels_from_old()

    def label_filter(label_id_st):
        label_names = []
        for label_id in label_id_st:
            if label_id == labels_dict[label_id] and label_id not in exc_list:
                label_id_st.remove(label_id)
            else:
                label_names.append(labels_dict[label_id])

        return ','.join(label_names)

    return label_id_series.str.split(',').apply(label_filter)
