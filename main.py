import logging
import re
import os.path
from time import time
from typing import List
from tabulate import tabulate

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from lib.read_mails import read_n_mails, store_n_mails
from lib.set_labels import label_mails, reset_labels, store_list_of_labels, create_labels, list_labels_from_old
from lib.ML import train_and_dump_model, FitModel, write_label_names


# setting up logger to see logs
TEST_MODE = False
READ_REAL_DATA = True
logging.basicConfig(
    level=logging.DEBUG if TEST_MODE else logging.ERROR,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

menu_message = """\
# MENU #   
    1: list first n mails
    2: add labels to first n mails
    3: add labels to only unread mails
    4: label specific mails
    5: reset added labels
    6: change maximum limit of mails to be read
    7: train model on your data
    8: use different model
    9: create labels (for new user)
    help | cls | exit
    (Enter full screen for best experience)
"""

help_dict = {
    1: '',
    2: '',
    3: '',
    4: '',
    5: '',
    6: '',
    7: '',
    8: '',
    9: ''
}

help_message_end = """
cls- clear the screen
exit- to exit the application

help - to get help for a particular option 
        enter help and option no.
        eg. help 1 (to view help of 1st option)
"""

help_message = ''

for option in help_dict:
    help_message += str(option) + ': ' + help_dict[option] + '\n'

help_message += help_message_end


def read_mails(n):
    if READ_REAL_DATA:
        return read_n_mails(n)

    else:
        # temporary function for test purposes
        df = pd.read_csv('data/training_data.csv', sep='~', index_col=0, nrows=n)
        return df


def label_first_n_mails(n):
    with console.status('[bold green]Working...') as status:
        mails_df = read_mails(n)
        console.log('[green]Finished reading mails!')
        label_mails(mails_df)
        console.log('[green]Finished labeling mails!')
        console.log('[red]Done!')


def display_mails(n):
    with console.status(f'[bold green]Reading {n} mails!'):
        mail_df = read_mails(n)
        display_mail_df = mail_df.loc[:, ['sender', 'subject']]
        display_mail_df['label names'] = write_label_names(mail_df['labels'], exc_list=['UNREAD', 'INBOX'])

        # setting up word limit for display of subject line
        display_mail_df['subject'] = display_mail_df['subject'].apply(lambda text: re.sub('[^a-zA-Z0-9,.]', ' ', text)[:60])
        console.print(tabulate(display_mail_df, headers='keys', tablefmt='psql'))


def label_specific_mails(n: int, indices: List[int], resetLabels=False):
    with console.status('[bold green]Working...!'):
        mail_df = read_mails(n)
        specific_mails_df = mail_df.iloc[indices]
        if resetLabels:
            reset_labels(specific_mails_df)
        else:
            label_mails(specific_mails_df)
    console.log('[red]Done!')


def label_unread_mails(n: int):
    """
    labels unread mails out of first n mails
    """
    with console.status('[bold green]Working...!'):
        mails_df = read_mails(n)
        unread_mails_df = mails_df.loc[mails_df['labels'].str.contains('UNREAD', case=True)]
        label_mails(unread_mails_df)
    console.log('[red]Done!')


def store_user_data():
    """
    get user's training data and train model on that
    the data should be properly labeled
    """
    with console.status('Fetching labels'):
        store_list_of_labels()
        console.log('Labels stored')

    n_train_mails = int(console.input('Enter total no of mails in your mailbox: '))

    t0 = time()
    with console.status(f'Reading mails...'):
        console.log('[blue]It takes about 5-6 minutes per 1000 mails')
        store_n_mails(n_train_mails)
        console.log(f'[yellow]Read and stored mails in {time()-t0} seconds')


def train_model():
    t1 = time()
    model_name = console.input('which model to be fitted? svm/rf/knn: ')
    with console.status('Training model'):
        train_and_dump_model(model_name)
        console.log(f'[yellow]Model trained in {time()-t1} seconds')


label_name_list = [
    value for key, value in list_labels_from_old().items() if re.match('Label_[0-9]', key)
]

console = Console()
console.print(menu_message)

while True:
    limit_input = console.input('enter maximum mails to be handled: ')
    if limit_input.isdigit():
        max_mails_limit = int(limit_input)
        break
    else:
        console.print('[red] please enter integer value')

n = max_mails_limit
# with console.status('[green]fetching mails..'):
#     cached_mails_df = read_mails(max_mails_limit)

while True:
    input_msg = console.input('[bold green]gmail_organizer >> ')

    if input_msg == '1':
        n = int(console.input('enter no of mails: '))
        if n > max_mails_limit:
            logger.error('n is more than the max limit. pls change the limit')
            continue
        display_mails(n)

    elif input_msg == '2':
        n = int(console.input('enter no of mails: '))
        if n > max_mails_limit:
            logger.error('n is more than the max limit. pls change the limit')
            continue
        label_first_n_mails(n)
        display_mails(n)

    elif input_msg == '3':
        label_unread_mails(max_mails_limit)

    elif input_msg == '4':
        indices_str = console.input('enter indices: ')
        indices = [int(i) for i in indices_str.split(' ')]
        label_specific_mails(max_mails_limit, indices)

    elif input_msg == '5':
        indices_str = console.input('enter indices: ')
        indices = [int(i) for i in indices_str.split(' ')]
        label_specific_mails(max_mails_limit, indices, resetLabels=True)

    elif input_msg == '6':
        max_mails_limit = int(console.input('enter new limit: '))

    elif input_msg == '7':
        input_msg = console.input('Have you created the labels (yes/no)? ')
        if input_msg != 'yes':
            console.print('please create labels using option 9!')
            continue
        store_user_data()
        train_model()

    elif input_msg == '8':
        train_model()

    elif input_msg == '9':
        if not any(label_name_list):
            create_labels()
        else:
            logger.warning('you already have some labels set up!')
            console.log('''[red]Any old label with same name will be left as it is.
            but they will be used to label mails.
            other old labels will have no effect''')
            confirmation = console.input('do you still want to continue and create the labels (yes/no)? ')

            if confirmation == 'yes':
                with console.status('Creating labels...'):
                    create_labels()
                console.log('[red]Done')

    elif input_msg == 'menu':
        console.print(menu_message)

    elif input_msg == 'cls':
        console.clear()
        console.print(menu_message)

    elif input_msg == 'help':
        console.print(help_message)

    elif re.match('help [1-6]', input_msg):
        console.print(help_dict[int(input_msg[-1])])

    elif input_msg == 'exit':
        break

    else:
        logger.error('wrong option. refer to the help')
