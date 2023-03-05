import numpy as np
import pandas as pd
import joblib
from typing import List
import logging
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV, \
    train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from conf.user_info import USER_EMAIL_ID
import json
import re

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self):
        with open('../../data/label_dict.txt', 'r') as file:
            label_str = file.read()

        self.labels_dict = json.loads(label_str.replace('\'', '\"'))
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

    def _encode_corpus(self, corpus: pd.Series) -> np.ndarray:
        self.tfidf = TfidfVectorizer(
            max_df=0.8,
            min_df=0.05
        )
        encoded_corpus = self.tfidf.fit_transform(corpus).toarray()
        return encoded_corpus

    def get_encoded_corpus(self, df: pd.DataFrame) -> pd.DataFrame:
        final_df = self._clean_email_df(df)
        return pd.DataFrame(self._encode_corpus(final_df['body']))
    
    def get_training_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        final_df = self._clean_email_df(df)
        encoded_corpus_df = pd.DataFrame(self._encode_corpus(final_df['body']))
        encoded_labels_df = pd.DataFrame(self._encode_labels(final_df['labels']))
        return encoded_corpus_df, encoded_labels_df


class GenerateLabels(Preprocess):
    def __init__(self, model_name='knn'):
        super().__init__()
        self.model = joblib.load(f'../../data/{model_name}_model.pkl')

    def generate_labels(self, read_mails: pd.DataFrame) -> List[List[str]]:
        """
        return a dataframe of labels for the given dataframe of mails
        """
        encoded_message = super().get_encoded_corpus(read_mails)
        encoded_labels = self._predict(encoded_message)
        labels_list = super().mlb.inverse_transform(encoded_labels)
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
        X, y = super().get_training_data(self.df)
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X, y)
        logger.info('saving model')
        joblib.dump(knn_clf, '../../data/knn_model.pkl')
        logger.info('model saved!')


df1 = pd.read_csv('../../data/training_data.csv', sep='~', index_col=0)
knn_model = FitModel(df1)
knn_model.knn_fit_and_dump()
