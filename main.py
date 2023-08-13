import logging
from datetime import datetime
from time import process_time
import click
# local modules
from utils import set_check_device
from data_operations import data_loader, performance_check
from train import train_model
from models import save_model, CurrentCNN, SimpleCNN, TestCNN
# create logger
logging.basicConfig(filename='compilation_log.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8', level=logging.INFO)
logger = logging.getLogger('process_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


@click.command()
@click.option('--model_name', type=str, default='simpleCNN', help='Name of model to run')
def compile_selected_cnn_model(model_name):
    dataloader_train, dataloader_test = data_loader()
    if model_name == 'currentCNN':
        model = CurrentCNN()
    elif model_name == 'test':
        model = TestCNN()
    else:
        model = SimpleCNN()

    # the reason to separate train() from model is to enable multiple models comparisons
    train_model(model, dataloader_train)
    save_model(model)
    performance_check(dataloader_test, model)


if __name__ == '__main__':
    # enable the timer to measure operation time
    start_time = datetime.now()
    t_start = process_time()
    logger.info('\n##########     A new process has started     ##########')
    set_check_device()
    compile_selected_cnn_model()
    # finish all process and end the timer
    t_simple_cnn_stop = process_time()
    end_time = datetime.now()
    logger.info(f'Elapsed time for Simple CNN compilation in seconds: {t_simple_cnn_stop-t_start}')
    logger.info('#########     Process completes successfully     ##########\n')
