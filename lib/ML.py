import gzip
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# get user's information (email id)
USER_EMAIL_ID = json.load(open('conf/user_info.json'))['USER_EMAIL_ID']
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# loading labels
if os.path.exists('data/label_dict.json'):
    with open('data/label_dict.json', 'r') as file:
        labels_dict = json.load(file)
else:
    logger.error('File labels_dict not found')


def preprocess_text(text):
    """
    removes any special character
    """
    text = text.lower()
    text = re.sub('[^a-zA-Z,.]', ' ', text)
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


def get_encoded_corpus_for_train(df: pd.DataFrame) -> csr_matrix:
    """
    convert corpus of words into tfidf vectorized matrix with vocabulary of the corpus
    as a feature and each message as a row
    """
    # checking for column names
    if not [col in df for col in ['sender', 'sender', 'body']]:
        logger.error('dataframe doesnt contain required column names')
        raise Exception

    # Creating Tfidf Vectorizers for all the 3 fields
    subject_tfidf = TfidfVectorizer(preprocessor=preprocess_text, max_df=0.9, min_df=0.005)
    body_tfidf = TfidfVectorizer(preprocessor=preprocess_text, max_df=0.8, min_df=0.01)
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
    dill.dump(vectorizers, gzip.open('data/TfidfVectorizers.pklz', 'wb'))
    return feature_matrix


def write_label_names(label_id_series: pd.Series, exc_list=[]) -> pd.Series:
    """
    :param exc_list: a list of label_ids among the default ones which are required
    will convert comma separated label_ids into label_names for a series of strings
    """

    def label_filter(label_id_st):
        label_names = []
        for label_id in label_id_st:
            if label_id == labels_dict[label_id] and label_id not in exc_list:
                label_id_st.remove(label_id)
            else:
                label_names.append(labels_dict[label_id])

        return ','.join(label_names)

    return label_id_series.str.split(',').apply(label_filter)


class Preprocess:
    def __init__(self):
        self.labels_dict = labels_dict
        self.all_labels = [value for key, value in self.labels_dict.items() if key != value]

        self.mlb = MultiLabelBinarizer(classes=self.all_labels)
        self.mlb.fit(self.all_labels)

    def _clean_email_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # to remove emails sent by the user himself
        condition = df['sender'] == f'{USER_EMAIL_ID}'
        df.drop(df[condition].index, inplace=True)
        df_reindexed = df.reset_index(drop=True)

        # filling null values with empty string
        imputer = SimpleImputer(strategy='constant', fill_value='')
        final_df = pd.DataFrame(imputer.fit_transform(df_reindexed), columns=df_reindexed.columns)
        return final_df

    def _encode_labels(self, label_id_series: pd.Series, strategy: str = 'mlb') -> np.ndarray:
        label_name_series = write_label_names(label_id_series)
        labels_array = [list(st.split(',')) for st in label_name_series]
        return self.mlb.transform(labels_array)

    def _encode_corpus(self, df: pd.DataFrame) -> csr_matrix:
        # checking for column names
        if not [col in df for col in ['sender', 'sender', 'body']]:
            logger.error('dataframe doesnt contain required column names')
            raise Exception

        # loading vectorizers
        if os.path.exists('data/TfidfVectorizers.pklz'):
            self.tfidf = dill.load(gzip.open('data/TfidfVectorizers.pklz', 'rb'))

        else:
            self.tfidf = None
            logger.error('tfidf not loaded. should use get_encoded_corpus_for_train')
            raise Exception

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
        feature_matrix = get_encoded_corpus_for_train(final_df)
        encoded_labels_df = self._encode_labels(final_df['labels'])
        return feature_matrix, encoded_labels_df


class GenerateLabels(Preprocess):
    def __init__(self, model_name='svm'):
        super().__init__()
        self.model = joblib.load(f'data/{model_name}_model.pkl')

    def generate_labels(self, read_mails: pd.DataFrame) -> List[Tuple[str]]:
        """
        return labels for the given dataframe of mails from previously trained model
        """
        encoded_message = super().get_encoded_corpus(read_mails)
        encoded_labels = self.model.predict(encoded_message)
        labels_list = self.mlb.inverse_transform(encoded_labels)
        return labels_list


class FitModel(Preprocess):
    def __init__(self, df: pd.DataFrame, model_name='svm'):
        super().__init__()
        self.data_tup = self.get_training_data(df)
        self.model_name = model_name

    def fit_and_dump(self):
        feature_matrix, encoded_labels = self.data_tup
        if self.model_name == 'knn':
            clf = KNeighborsClassifier()
        elif self.model_name == 'svm':
            clf = MultiOutputClassifier(SVC(random_state=42, class_weight='balanced'))
        elif self.model_name == 'rf':
            clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        else:
            logger.error('incorrect model name')
            raise ValueError

        logger.info('training model')
        t0 = time()
        clf.fit(feature_matrix, encoded_labels)
        joblib.dump(clf, f'data/{self.model_name}_model.pkl')
        logger.info(f'model saved! took {time() - t0} seconds')


def train_and_dump_model(model_name='svm'):
    """
    will train the model on training data and store the model
    """
    if os.path.exists('data/training_data.csv'):
        df1 = pd.read_csv('data/training_data.csv', sep='~', index_col=0)
        model = FitModel(df1, model_name)
        model.fit_and_dump()

    else:
        logger.error('training data does not exist yet')


if __name__ == '__main__':
    train_and_dump_model('svm')
