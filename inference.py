"""
The script to run inference for the test dataset
"""
import logging
from datetime import datetime
from time import process_time
import click
import threading
import torch
# local modules
from utils import set_check_device
from data_operations import data_loader
from exceptions import InferenceTimeAssertError
# create logger
logging.basicConfig(filename='inference_log.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8', level=logging.INFO)
logger = logging.getLogger('inference_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def model_inference(model, dataloder_data, avg_sec=20):
    # evaluation mode has no dropout, batch normalization, etc.
    model.eval()

    with torch.no_grad():
        start_time = process_time()
        for i, data in enumerate(dataloder_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            _ = model(inputs)
        # end_time = time.time()
        end_time = process_time()
        elapsed_time = end_time - start_time
        # criteria: average inference time per image ≤ 20ms on a single CPU thread
        inference_time_ms = elapsed_time / len(dataloder_data) * 1000   # total_time / num_of_images * convert to ms

        if inference_time_ms > avg_sec:
            raise InferenceTimeAssertError(f'Average inference time per image: {inference_time_ms:.4f}ms, '
                                           f'reduce {(inference_time_ms-avg_sec):.4f}ms')
        else:
            logger.info(f'Average Inference Time per Image: {inference_time_ms:.4f}ms')
        logger.info(f'Inference Time: {elapsed_time:.4f} seconds')


def single_threaded_inference(model, input_data, avg_sec=20):
    """Create a single-threaded environment"""
    torch.set_num_threads(1)
    try:
        model_inference(model, input_data, avg_sec)
    except InferenceTimeAssertError as e:
        logger.exception(f'Exceed allowed average inference time per image: {avg_sec}')


def run_inference(input_data, avg_sec):
    """Run inference within a single thread"""
    cnn = torch.load('outputs/simpleCNN.pth')
    thread = threading.Thread(target=single_threaded_inference, args=(cnn, input_data, avg_sec))
    thread.start()
    thread.join()


if __name__ == '__main__':
    # enable the timer to measure operation time
    logger.info('\n##########     A new process has started     ##########')
    t_start = process_time()
    set_check_device()
    dataloader_train, dataloader_test = data_loader()
    t_inf_start = process_time()
    run_inference(dataloader_test, avg_sec=20)
    # finish all process and end the timer
    t_stop = process_time()
    logger.info(f'Elapsed time for inference: {t_stop-t_inf_start:.2f} sec')
    logger.info(f'Elapsed time for data loading, inference: {t_stop-t_start:.2f} sec')
    logger.info('#########     Process completes successfully     ##########\n')
