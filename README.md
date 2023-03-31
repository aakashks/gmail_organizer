# GMail Organizer v2.0

## MultiOutput Classification of Gmails

A menu based command-line application that automatically applies labels in the user's Gmail using classification.

## Description

It uses Gmail API for reading and modifying gmail messages. Support Vector Classifier is used to perform 
the classification (with OvR), although some other classifiers like RandomForestClassifier also work fine.
For Natural Language Processing of email text, I have used term frequency - inverse document frequency (TF-IDF) Vectorizer for vectorizing the corpus.
It provides the advantage that 
> a high weight of the tf-idf calculation is reached when we have a high term frequency(tf) in the given document and a 
> low document frequency of the term in the whole collection.

The goal of developing this application was to let users quickly sort their emails, thus saving them time.
The model evaluation and other details are described in detail in [this notebook](notebooks/ML_supervised2.ipynb).

## Usage

The application can be set up very quickly, with the only tricky part being the Gmail API key generation.
For usage, please see the [user guide](USER_GUIDE.md)

## Challenges faced

I tried to use KMeans Clustering to cluster the email, but the performance was poor.
Due to this reason, I also discarded my plan to use semi-supervised ML.

All the work I did to attempt to use different ML techniques is described in the jupyter notebooks.

Also, as the size of the training dataset was small, I avoided using complex NLP and classification techniques as that 
wouldn't yield a huge improvement in performance.

## Limitations

The models were trained on a small dataset of ~1200 emails I labeled manually. 
I did not label every mail category, but only a few categories that seemed important to me.
As a result, the classifier also doesn't label many emails 
(as most of them are similar to those which were not important to me)

The whole application has limited features because of time constraints.

## Data Privacy

Your emails or personal data, whenever used, is always stored on your local storage only. \
Never upload sensitive files, especially **`credentials.json`**, **`token.json`**, and your Gmail API key.

## References
[GMail API reference](https://developers.google.com/gmail/api/reference/rest/v1/users.messages)\
[Rich library for colorful display](https://rich.readthedocs.io/en/stable/console.html)\
[TFIDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)