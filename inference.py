"""
The script to run inference for the test dataset
"""
import logging
import os.path
from time import process_time
import click
import threading
import torch
# local modules
from utils import set_check_device
from data_operations import data_loader
from exceptions import InferenceTimeAssertError
# create logger
logging.basicConfig(filename='log_inference.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8', level=logging.INFO)
logger = logging.getLogger('inference_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


@click.command()
@click.option('--model_file', type=str, default='outputs/SimpleCNN/SimpleCNN.pth', help='Model file to run')
@click.option('--avg_sec', type=int, default=20, help='Average inference time threshold per image in ms')
@click.option('--n_thread', type=int, default=1, help='Number of thresholds to run')
def run_inference(model_file, avg_sec, n_thread):
    """Run inference within a single thread"""
    model = torch.load(model_file)
    thread = threading.Thread(target=threaded_inference, args=(model, avg_sec, n_thread))
    thread.start()
    thread.join()


def model_inference(model, avg_sec=20):
    if model.is_grayscale:
        _, dataloader_data = data_loader(n_channel=1)
    else:
        _, dataloader_data = data_loader(n_channel=3)
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        infer_start = process_time()
        for i, data in enumerate(dataloader_data, 0):
            inputs, labels = data
            _ = model(inputs)
        infer_end = process_time()
        elapsed_time = infer_end - infer_start
        # criteria: average inference time per image ≤ 20ms on a single CPU thread
        inference_time_ms = elapsed_time / len(dataloader_data) * 1000   # total_time / num_of_images * convert to ms
        if inference_time_ms > avg_sec:
            raise InferenceTimeAssertError(f'Average inference time per image: {inference_time_ms:.4f}ms, '
                                           f'reduce {(inference_time_ms-avg_sec):.4f} ms')
        else:
            logger.info(f'Average Inference Time per Image: {inference_time_ms:.4f} ms')
        logger.info(f'Inference Time: {elapsed_time:.4f} seconds')


def threaded_inference(model, avg_sec=20, n_thread=1):
    """Create a single-threaded environment"""
    torch.set_num_threads(n_thread)
    try:
        model_inference(model, avg_sec)
    except InferenceTimeAssertError as e:
        logger.exception(f'Exceed allowed average inference time per image: {avg_sec}')


if __name__ == '__main__':
    # enable the timer to measure operation time
    logger.info('\n##########     A new process has started     ##########')
    t_start = process_time()
    set_check_device()
    run_inference()
    # finish all process and end the timer
    t_stop = process_time()
    logger.info(f'Elapsed time for data loading, inference: {t_stop-t_start:.2f} sec')
    logger.info('#########     Process completes successfully     ##########\n')
