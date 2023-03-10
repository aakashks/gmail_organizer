# GMail Organizer

## Multioutput Classification of mails

Using machine learning to classify Gmail mails into different labels can be of practical use, even though Gmail already has a feature for creating filters. While Gmail's filtering feature is useful, it requires manual configuration by the user and can be time-consuming, especially when dealing with large volumes of email.

Machine learning, on the other hand, can automatically classify emails into different labels based on patterns it identifies in the content, sender, and other metadata of the emails. This can save time and effort for users, particularly those who receive large volumes of email and need to sort them quickly and efficiently.

Additionally, machine learning can potentially provide more accurate and personalized results than filters created by the user, as it can learn from the user's past behavior and preferences. For example, if the user often moves emails from a particular sender to a specific label, the machine learning model can learn this pattern and automatically apply the label to future emails from that sender.

Overall, while Gmail's filtering feature is useful, machine learning can provide additional benefits such as automation, accuracy, and personalization, making it a practical and valuable tool for email classification.

Also, many of IITR's mails are marked as important even though there are much less relevant. This issue will also be coped as only the emails preferred by the user will be marked as IMP(which is a separate label for important mails).

## Usage

For Usage please see the [user guide](USER_GUIDE.md)

## Limitations

1. The Model has currently been trained on my IITR mailbox which had ~991 mails at that time.
Given that I had quickly labeled only some particular categories of mails, the model gives predictions according to that only.
Hence, one might observe that many mails will not be labeled by the model as they were
similar to those mails which were not important for me and had not been labeled by me.
Feature for active/ online learning might be added in the future.
2. Currently only the body of any email is considered for classification.
3. Some mails cannot be read as their parsed json data format doesn't match
one of the predefined ones.
4. Due to some reasons many features work exclusively on IITR gmail accounts
and some functionalities are restricted in other gmail accounts.

## References
[GMail API reference](https://developers.google.com/gmail/api/reference/rest/v1/users.messages)\
[Rich library for colourful display](https://rich.readthedocs.io/en/stable/console.html)\
[TFIDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
