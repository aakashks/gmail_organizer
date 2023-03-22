import dill
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
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

# get user's information (email id)
USER_EMAIL_ID = json.load(open('conf/user_info.json'))['USER_EMAIL_ID']
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    removes any special character
    """
    text = text.lower()
    text = re.sub('[^a-zA-Z,]', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return processed_text


def preprocess_sender(address):
    """
    will separate domain name from the sender's address
    also separate department so that it has effect in the model
    """
    address_lst = address.lower().split('@')
    address_lst[1] = re.sub('[.]ac|[.]in|[.]com', '', address_lst[1])
    address_lst[1] = re.sub('[.]', ' ', address_lst[1])
    address_lst[0] = re.sub('[._]', '', address_lst[0])
    return ' '.join(address_lst)


def get_encoded_corpus_for_train(df: pd.DataFrame, max_df=0.95, min_df=0.05) -> csr_matrix:
    """
    convert corpus of words into tfidf vectorized matrix with vocabulary of the corpus
    as a feature and each message as a row
    """
    # checking for column names
    if not [col in df for col in ['sender', 'sender', 'body']]:
        logger.error('dataframe doesnt contain required column names')
        raise Exception

    # Creating Tfidf Vectorizers for all the 3 fields
    subject_tfidf = TfidfVectorizer(preprocessor=preprocess_text, min_df=0.01)
    body_tfidf = TfidfVectorizer(preprocessor=preprocess_text, max_df=max_df, min_df=min_df)
    sender_tfidf = TfidfVectorizer(preprocessor=preprocess_sender)

    # fitting and transforming the respective features of dataframe into sparse matrices
    subject_vectors = subject_tfidf.fit_transform(df['subject'])
    body_vectors = body_tfidf.fit_transform(df['body'])
    sender_vectors = sender_tfidf.fit_transform(df['sender'])

    # concatenating sparse matrices
    feature_matrix = hstack((subject_vectors, body_vectors, sender_vectors))

    # dump tfidf vectorizers to reuse vocabulary
    vectorizers = [sender_tfidf, body_tfidf, subject_tfidf]
    # using dill as it will also serialize the user defined preprocessing functions
    dill.dump(vectorizers, open('data/TfidfVectorizers.pkl', 'wb'))
    return feature_matrix


class Preprocess:
    def __init__(self):
        self.mlb = joblib.load('data/multiLabelBinarizer.pkl')
        # checking if file exists or not
        if os.path.exists('data/TfidfVectorizers.pkl'):
            self.tfidf = dill.load(open('data/TfidfVectorizers.pkl', 'rb'))

        else:
            self.tfidf = None

    def _clean_email_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # to remove emails sent by the user himself
        condition = df['sender'] == f'{USER_EMAIL_ID}'
        df.drop(df[condition].index, inplace=True)
        df_reindexed = df.reset_index(drop=True)

        # filling null values with empty string
        imputer = SimpleImputer(strategy='constant', fill_value='')
        final_df = pd.DataFrame(imputer.fit_transform(df_reindexed), columns=df_reindexed.columns)
        return final_df

    def _encode_labels(self, label_series: pd.Series, strategy: str = 'mlb') -> np.ndarray:
        labels_array = [list(st.split(',')) for st in label_series]
        return self.mlb.transform(labels_array)

    def _encode_corpus(self, df: pd.DataFrame) -> csr_matrix:
        # checking for column names
        if not [col in df for col in ['sender', 'sender', 'body']]:
            logger.error('dataframe doesnt contain required column names')
            raise Exception

        # unpacking loaded tfidf
        if self.tfidf:
            logger.error('tfidf not loaded. should use get_encoded_corpus_for_train')
            raise ValueError

        sender_tfidf, subject_tfidf, body_tfidf = self.tfidf

        # fitting and transforming the respective features of dataframe into sparse matrices
        subject_vectors = subject_tfidf.transform(df['subject'])
        body_vectors = body_tfidf.transform(df['body'])
        sender_vectors = sender_tfidf.transform(df['sender'])

        # concatenating sparse matrices
        feature_matrix = hstack((subject_vectors, body_vectors, sender_vectors))

        return feature_matrix

    def get_encoded_corpus(self, df: pd.DataFrame) -> csr_matrix:
        final_df = self._clean_email_df(df)
        return self._encode_corpus(final_df)

    def get_training_data(self, df: pd.DataFrame) -> Tuple[csr_matrix, np.ndarray]:
        final_df = self._clean_email_df(df)
        encoded_corpus_df = get_encoded_corpus_for_train(final_df)
        encoded_labels_df = self._encode_labels(final_df['labels'])
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
        encoded_labels = self.model.predict(encoded_message)
        labels_list = self.mlb.inverse_transform(encoded_labels)
        return labels_list


class FitModel(Preprocess):
    def __init__(self, df: pd.DataFrame, model_name='knn'):
        super().__init__()
        self.df = df

    def knn_fit_and_dump(self):
        encoded_message_body, encoded_labels = self.get_training_data(self.df)
        knn_clf = KNeighborsClassifier()
        logger.info('training model')
        t0 = time()
        knn_clf.fit(encoded_message_body, encoded_labels)
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
