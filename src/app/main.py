from time import time

from src.lib.ML import GenerateLabels
from src.lib.read_mails import read_n_mails
import logging

# setting up logger to see logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('main program started')
    t0 = time()
    sample_mails = read_n_mails(50)
    knn_labeler = GenerateLabels()
    print(knn_labeler.generate_labels(sample_mails))
    logger.info(f'program finished in {time()-t0} seconds')
