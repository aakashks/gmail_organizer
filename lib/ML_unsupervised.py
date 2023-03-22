import json
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# get user's information (email id)
with open('conf/user_info.json') as file:
    USER_EMAIL_ID = json.load(file)['USER_EMAIL_ID']


def preprocess_data(mails_df: pd.DataFrame):
    """
    :return: scipy sparse matrix
    """
    condition = mails_df['sender'] == f'{USER_EMAIL_ID}'
    mails_df.drop(mails_df[condition].index, inplace=True)
    mails_df = mails_df.reset_index(drop=True)
    imputer = SimpleImputer(strategy='constant', fill_value='')
    mails_df = pd.DataFrame(imputer.fit_transform(mails_df), columns=mails_df.columns)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        """
        removes any special character
        """
        text = text.lower()
        text = re.sub('[^a-zA-Z]', ' ', text)
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

    # Transform particular columns into vectors
    preprocessor = ColumnTransformer(transformers=[
        ('subject', TfidfVectorizer(preprocessor=preprocess_text, min_df=0.01), 'subject'),
        ('body', TfidfVectorizer(preprocessor=preprocess_text, max_df=0.9, min_df=0.1), 'body'),
        ('sender', TfidfVectorizer(preprocessor=preprocess_sender), 'sender')
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    return pipeline.fit_transform(mails_df)


def cluster_mails(preprocessed_data) -> np.ndarray:

    k = 25
    model = KMeans(n_clusters=k, random_state=42)
    predicted_labels = model.fit_predict(preprocessed_data)
    return predicted_labels


def propagate_labels(original_labels, predicted_labels_int, n_clusters):
    propagated_labels = pd.Series([], name='propagated_labels', dtype='object')
    return


