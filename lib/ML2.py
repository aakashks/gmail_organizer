import json
import logging
import os.path
import re
from time import time
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return processed_text


class Preprocess:
    def __init__(self):
        with open('conf/user_info.json') as file:
            self.USER_EMAIL_ID = json.load(file)['USER_EMAIL_ID']
        if os.path.exists('data/label_dict.json'):
            with open('data/label_dict.json', 'r') as file:
                self.labels_dict = json.load(file)
        else:
            logger.error('File labels_dict not found')

        self.all_labels = [key for key in self.labels_dict.keys() if re.match('Label_[0-9]', key)]

        self.mlb = MultiLabelBinarizer(classes=self.all_labels)
        self.mlb.fit(self.all_labels)
        self.preprocessor = ColumnTransformer(transformers=[
            ('subject', TfidfVectorizer(preprocessor=preprocess_text, min_df=0.1), 'subject'),
            ('body', TfidfVectorizer(preprocessor=preprocess_text, max_df=0.9, min_df=0.05), 'body'),
            ('sender', TfidfVectorizer(ngram_range=(1, 2), lowercase=False), 'sender')
        ], remainder='drop')
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('preprocessor', self.preprocessor),
            ('tsvd', TruncatedSVD(n_components=50))
        ])

    def _clean_email_df(self, df: pd.DataFrame) -> pd.DataFrame:
        condition = df['sender'] == f'{self.USER_EMAIL_ID}'
        df_reindexed = df.drop(df[condition].index).reset_index(drop=True)
        return df_reindexed

    def _encode_labels(self, label_series: pd.Series, strategy: str = 'mlb') -> np.ndarray:
        labels_array = [list(st.split(',')) for st in label_series]
        return self.mlb.transform(labels_array)

    def get_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        final_df = self._clean_email_df(df)
        encoded_corpus_df = self.pipeline.fit_transform(final_df)
        encoded_labels_df = pd.DataFrame(self._encode_labels(final_df['labels']))
        return encoded_corpus_df, encoded_labels_df


class GenerateLabels(Preprocess):
    def __init__(self, model_name='knn'):
        super().__init__()
        self.model = joblib.load(f'data/{model_name}_model.pkl')
        self.preprocess_text = joblib.load(f'data/preprocess_text.pkl')
        self.pipeline = joblib.load(f'data/pipeline.pkl')

    def generate_labels(self, read_mails: pd.DataFrame) -> List[Tuple[str]]:
        """
        return labels for the given dataframe of mails
        """
        mails_df = super()._clean_email_df(read_mails)
        encoded_message = self.pipeline.transform(mails_df)
        encoded_labels = self.model.predict(encoded_message)
        labels_list = self.mlb.inverse_transform(encoded_labels)
        return labels_list


class FitModel(Preprocess):
    def __init__(self, df: pd.DataFrame, model_name='knn'):
        super().__init__()
        self.df = df

    def knn_fit_and_dump(self):
        encoded_message_body_df, encoded_labels_df = self.get_training_data(self.df)
        joblib.dump(self.pipeline, 'data/pipeline.pkl')
        joblib.dump(preprocess_text, 'data/preprocess_text.pkl')
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