import requests

SUCCESS = True
FAILURE = False


def post_to_add_label(msg_id, label_id_list):
    label_json = {"addLabelIds": label_id_list}
    r = requests.post(
        fr'POST https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}/modify',
        json=label_json
    )
    if r.status_code == 200:
        return SUCCESS
    else:
        return FAILURE

