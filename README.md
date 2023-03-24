# GMail Organizer v2.0

## MultiOutput Classification of Gmails

A command line application which automatically apply labels in the user's gmail using classification.

## Description

It uses Gmail API for reading and modifying gmail messages. Support Vector Classifier is used to perform 
the classification (other classifiers can also be used easily like- KNearest Neighbours).
Also, I have used term frequency - inverse document frequency (TF-IDF) for vectorizing words.
It provides the advantage that a high weight of the tf-idf calculation is reached when we have 
a high term frequency(tf) in the given document and a low document frequency of the term in the whole collection

The goal of developing this application was to let user easily sort his emails, thus saving his time.

## Usage

The application can be set up very easily, with the only difficult part being the gmail api key generation.
For usage please see the [user guide](USER_GUIDE.md)

## Challenges faced

I tried to use KMeans Clustering to cluster the email, but the performance was poor.
Due to this reason I also discarded my plan to use semi-supervised ML.

All the work I did to attempt to use different ML techniques are mentioned in the jupyter notebooks.

Also, as the size of the training dataset was small, I avoided using complex NLP and classification
techniques as that wouldn't yield a huge improvement in performance.

## Limitations

The models were trained on a small dataset of ~1200 mails labelled manually by me. 
I did not label every category of mails, but only few categories which seemed to be important to me.
As a result, the classifier also doesn't label many mails (as most of them are similar to those
which were not important for me)

The whole application has limited features because of the limited time in which it was developed.

## Data Privacy

Your gmail or any personal data whenever used, is always stored on your local storage only. \
Never upload sensitive files especially, **`credentials.json`**, **`token.json`** and your gmail API key.

## References
[GMail API reference](https://developers.google.com/gmail/api/reference/rest/v1/users.messages)\
[Rich library for colourful display](https://rich.readthedocs.io/en/stable/console.html)\
[TFIDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)