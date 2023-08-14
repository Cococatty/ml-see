"""
The central caller to run training and evaluation for the chosen model
Available Options as Inputs:
- mode
- model_name
- n_channel
"""
import logging
from time import process_time
import click
# local modules
from utils import set_check_device
from data_operations import data_loader
from train import train_model, evaluate_model
from models import save_model, SimpleCNN, SimpleCNN_GrayScale
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
@click.option('--model_name', type=str, default='SimpleCNN', help='Name of model to run, choose from SimpleCNN, SimpleCNN_GrayScale')
@click.option('--n_channel', type=int, default=3, help='Number of channels to run model with') 
def compile_selected_cnn_model(mode, model_name, n_channel):
    model = SimpleCNN_GrayScale() if n_channel == 1 else SimpleCNN()
    dataloader_train, dataloader_test = data_loader(n_channel=n_channel)
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
