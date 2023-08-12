import logging
# create logger
logging.basicConfig(filename='compilation_log.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8', level=logging.INFO)
logger = logging.getLogger('process_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
from datetime import datetime
from time import process_time
# local modules
from utils import set_check_device
from data_operations import data_loader
from train import train_model
from models import SimpleCNN
from inference import performance_check


def compile_simple_cnn_model():
    dataloader_train, dataloader_test = data_loader()
    cnn = SimpleCNN()
    # the reason to separate train() from model is to enable multiple models comparisons
    train_model(cnn, dataloader_train)
    cnn.save_model()
    performance_check(dataloader_test=dataloader_test, model=cnn)
    # inference
    # validation
    # performance review


if __name__ == '__main__':
    # Get the current datetime object
    start_time = datetime.now()
    t_start = process_time()
    logger.info('\n##########     A new process has started     ##########')
    set_check_device()
    compile_simple_cnn_model()
    t_simple_cnn_stop = process_time()
    end_time = datetime.now()
    logger.info(f'Elapsed time for Simple CNN compilation in seconds: {t_simple_cnn_stop-t_start}')
    logger.info('#########     Process is completed successfully     ##########\n')
