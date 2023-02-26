from src.lib.read_mails import store_all_mails
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('main program started')
    store_all_mails(True)
    logger.info('end program')
