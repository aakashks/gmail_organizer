import re

import numpy as np
import pandas as pd

from src.lib.set_labels import label_mails, reset_labels
from src.lib.read_mails import read_n_mails
import logging
from typing import List
from rich import print as rprint
from rich.logging import RichHandler
from rich.console import Console

# setting up logger to see logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

menu_message = """\
MENU:
    1: add labels to first n mails
    2: list first n mails
    3: add labels to only unread mails
    4: label specific mails
    5: reset added labels
    6: change maximum limit of mails to be read
}
"""

help_dict = {
    1: '',
    2: '',
    3: """add labels to only unread mails
    will check only first 500 mails for unread status
    """,
    4: '',
    5: '',
    6: ''
}

help_message = """
cls- clear the screen
exit- to exit the application
"""

max_mails_limit = 500


def label_first_n_mails(n):
    with console.status('[bold green]Working...') as status:
        mails_df = read_n_mails(n)
        console.log('[green]Finished reading mails!')
        label_mails(mails_df)
        console.log('[green]Finished labeling mails!')
        console.log('[red]Done!')


def display_mails(n):
    with console.status(f'[bold green]Reading {n} mails!'):
        mail_df = read_n_mails(n)
        mail_df_relevant = mail_df.loc[:, ['sender', 'subject', 'labels']]
        console.print(mail_df_relevant)


def label_specific_mails(n: int, indices: List[int], resetLabels=False):
    mail_df = read_n_mails(n)
    specific_mails_df = mail_df.iloc[indices]
    if resetLabels:
        reset_labels(specific_mails_df)
    else:
        label_mails(specific_mails_df)


def label_unread_mails(n: int):
    """
    labels unread mails out of first n mails
    """
    mails_df = read_n_mails(n)
    unread_mails_df = mails_df.loc[mails_df['labels'].str.contains('UNREAD', case=True)]
    label_mails(unread_mails_df)


console = Console()
console.print(menu_message)
n = max_mails_limit

while True:
    input_msg = console.input('gmail_organizer $ ')

    if input_msg == '1':
        n = int(console.input('enter no of mails: '))
        if n > max_mails_limit:
            logger.error('n is more than the max limit. pls change the limit')
            continue
        label_first_n_mails(n)

    elif input_msg == '2':
        n = int(console.input('enter no of mails: '))
        if n > max_mails_limit:
            logger.error('n is more than the max limit. pls change the limit')
            continue
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
