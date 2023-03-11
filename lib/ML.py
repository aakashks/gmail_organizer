import json
import logging
import re
import os.path
from time import time
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

# get user's information (email id)
with open('conf/user_info.json') as file:
    USER_EMAIL_ID = json.load(file)['USER_EMAIL_ID']


def _encode_corpus_for_train(corpus: pd.Series, max_df=0.8, min_df=0.05) -> np.ndarray:
    """
    convert corpus of words into tfidf vectorized matrix with vocabulary of the corpus
    as a feature and each message as a row
    """
    tfidf = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df
    )
    encoded_corpus = tfidf.fit_transform(corpus).toarray()
    joblib.dump(tfidf.vocabulary_, 'data/tfidf_vocabulary.pkl')
    return encoded_corpus


class Preprocess:
    def __init__(self):
        if os.path.exists('data/label_dict.json'):
            with open('data/label_dict.json', 'r') as file:
                self.labels_dict = json.load(file)
        else:
            logger.error('File labels_dict not found')

        self.all_labels = [key for key in self.labels_dict.keys() if re.match('Label_[0-9]', key)]

        self.mlb = MultiLabelBinarizer(classes=self.all_labels)
        self.mlb.fit(self.all_labels)

    def _clean_email_df(self, df: pd.DataFrame) -> pd.DataFrame:
        condition = df['sender'] == f'{USER_EMAIL_ID}'
        df.drop(df[condition].index, inplace=True)
        df_reindexed = df.reset_index(drop=True)
        imputer = SimpleImputer(strategy='constant', fill_value='')
        final_df = pd.DataFrame(imputer.fit_transform(df_reindexed), columns=df_reindexed.columns)
        return final_df

    def _encode_labels(self, label_series: pd.Series, strategy: str = 'mlb') -> np.ndarray:
        labels_array = [list(st.split(',')) for st in label_series]
        return self.mlb.transform(labels_array)

    def _encode_corpus(self, message_body: pd.Series) -> np.ndarray:
        vocab = joblib.load('data/tfidf_vocabulary.pkl')
        self.tfidf = TfidfVectorizer(vocabulary=vocab)
        encoded_corpus = self.tfidf.fit_transform(message_body).toarray()
        return encoded_corpus

    def get_encoded_corpus(self, df: pd.DataFrame) -> pd.DataFrame:
        final_df = self._clean_email_df(df)
        return pd.DataFrame(self._encode_corpus(final_df['body']))

    def get_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        final_df = self._clean_email_df(df)
        encoded_corpus_df = pd.DataFrame(_encode_corpus_for_train(final_df['body']))
        encoded_labels_df = pd.DataFrame(self._encode_labels(final_df['labels']))
        return encoded_corpus_df, encoded_labels_df


class GenerateLabels(Preprocess):
    def __init__(self, model_name='knn'):
        super().__init__()
        self.model = joblib.load(f'data/{model_name}_model.pkl')

    def generate_labels(self, read_mails: pd.DataFrame) -> List[Tuple[str]]:
        """
        return labels for the given dataframe of mails
        """
        encoded_message = super().get_encoded_corpus(read_mails)
        encoded_labels = self._predict(encoded_message)
        labels_list = self.mlb.inverse_transform(encoded_labels)
        return labels_list

    def _predict(self, encoded_message):
        """
        return encoded labels for the given encoded message
        """
        encoded_labels = self.model.predict(encoded_message)
        return encoded_labels


class FitModel(Preprocess):
    def __init__(self, df: pd.DataFrame, model_name='knn'):
        super().__init__()
        self.df = df

    def knn_fit_and_dump(self):
        encoded_message_body_df, encoded_labels_df = self.get_training_data(self.df)
        knn_clf = KNeighborsClassifier()
        logger.info('training model')
        t0 = time()
        knn_clf.fit(encoded_message_body_df, encoded_labels_df)
        joblib.dump(knn_clf, 'data/knn_model.pkl')
        logger.info(f'model saved! took {time() - t0} seconds')


def train_and_dump_model():
    """
    will train the model on training data and store the model
    """
    if os.path.exists('data/training_data.csv'):
        df1 = pd.read_csv('data/training_data.csv', sep='~', index_col=0)
        knn_model = FitModel(df1)
        knn_model.knn_fit_and_dump()

    else:
        logger.error('training data does not exist yet')
