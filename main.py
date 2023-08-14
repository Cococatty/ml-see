import logging
from time import process_time
import click
# local modules
from utils import set_check_device
from data_operations import data_loader
from train import train_model, evaluate_model
from models import save_model, SimpleCNN, TestCNN
# create logger
logging.basicConfig(filename='log_compilation.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8', level=logging.INFO)
logger = logging.getLogger('process_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


@click.command()
@click.option('--mode', type=str, default='compile', help='Compiling mode, chose from test, compile')
@click.option('--model_name', type=str, default='test', help='Name of model to run')  # simpleCNN
def compile_selected_cnn_model(mode, model_name):
    if mode == 'test':
        # model = TestCNN()
        model = CNNWithDropout()
    else:
        dataloader_train, dataloader_test = data_loader()
        if model_name == 'SimpleCNN':
            model = SimpleCNN()
        else:
            model = CNNWithDropout()
        # separate train() from model to enable multiple models comparisons
        train_model(model, dataloader_train)
        save_model(model)
        del dataloader_train
        
        evaluate_model(dataloader_test, model)
        del dataloader_test, model


if __name__ == '__main__':
    # enable the timer to measure operation time
    t_start = process_time()
    logger.info('\n##########     A new process has started     ##########')
    set_check_device()
    compile_selected_cnn_model()
    logger.info(f'Elapsed time for Simple CNN compilation in seconds: {process_time()-t_start}')
    logger.info(f'##########     Process completes successfully     ##########\n')
