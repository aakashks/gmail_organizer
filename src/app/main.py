from src.lib.set_labels import label_first_n_mails
import logging

# setting up logger to see logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    label_first_n_mails(20)
