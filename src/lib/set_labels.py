from src.lib.authorize import build_service
import pickle

service = build_service()


def list_labels():
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        return []

    label_dict = {label['id']: label['name'] for label in labels}
    return label_dict


# all_labels = str(list_labels())
#
# with open('../../data/label_dict.txt', 'w') as file:
#     file.write(all_labels)


def list_labels2():
    with open('../../data/label_dict.pickle', 'rb') as file:
        labels_dict = pickle.load(file)
    return labels_dict


def create_labels(): ...


def set_label(): ...


def get_unread_mails(): ...