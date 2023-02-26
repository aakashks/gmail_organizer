from src.lib.read_mails import store_n_mails
import logging

# setting up logger to see logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('main program started')
    store_n_mails(501)
    logger.info('end program')
