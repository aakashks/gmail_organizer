from src.lib.read_mails import store_all_mails
import logging

# setting up logger to see logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('main program started')
    store_all_mails(51, 10, test_mode=True)
    logger.info('end program')
