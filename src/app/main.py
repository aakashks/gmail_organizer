import numpy as np
import pandas as pd

from src.lib.set_labels import label_mails
from src.lib.read_mails import read_n_mails
import logging
from rich import print as rprint
from rich.logging import RichHandler
from rich.console import Console

console = Console()

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
    1. add labels to first n mails
    2. list first n mails
    3. add labels to unread mails out of first n mails
"""

help_message = """\
"""


def label_first_n_mails(n):
    with console.status('[bold green]Working...') as status:
        mails_df = read_n_mails(n)
        console.log('[green]Finished reading mails!')
        label_mails(mails_df)
        console.log('[green]Finished labeling mails!')
        console.log('[red]Done!')


def render_mails(n):
    with console.status(f'[bold green]Reading {n} mails!'):
        mail_df = read_n_mails(n)
        mail_df_relevant = mail_df.loc[:, ['sender', 'subject']]
        console.print(mail_df_relevant)


rprint(menu_message)

while True:
    choice = input('gmail_organizer $ ')

    if choice == '1':
        n = int(input('\tenter no of mails: '))
        label_first_n_mails(n)

    elif choice == '2':
        n = int(input('\tenter no of mails: '))
        render_mails(n)

    elif choice == 'exit':
        break
    else:
        logger.error('wrong option. refer to the help')
