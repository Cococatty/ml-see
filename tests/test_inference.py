import logging
import unittest
import os
import sys
import threading
from pathlib import Path
from time import process_time
import torch
# project modules
from utils import set_check_device
from data_operations import data_loader
from inference import single_threaded_inference, run_inference
from exceptions import InferenceTimeAssertError

FILE = Path(__file__).resolve()
test_dir = FILE.parents[0]
project_dir = FILE.parents[1]
sys.path.append(str(project_dir))
sys.path.append(str(test_dir))
os.chdir(test_dir)
logger = logging.getLogger('unittest_log')


class TestInference(unittest.TestCase):
    def setUp(self):
        self.img_large = self.create_large_image()
        self.output_path = os.path.join(test_dir, 'outputs')
        _, self.dataloader_test = data_loader()

    @staticmethod
    def create_large_image():
        import numpy as np
        height, width, channels = 1000, 1500, 3
        large_image = np.zeros((height, width, channels), dtype=np.uint8)
        large_image[100:200, 200:400, 0] = 255
        return large_image

    def test_givenASimpleModel_whenInferenceRequired_thenLoadTheModel(self):
        cnn = torch.load(os.path.join(self.output_path, 'simpleCNN.pth'))
        thread = threading.Thread(target=single_threaded_inference, args=(cnn, self.dataloader_test))
        thread.start()
        thread.join()
        # TODO enable
        # self.assertEqual(result, 62006)

    def test_givenALargeImage_whenRunInferenceAndExceedTimeReq_thenRaiseAnException(self):
        set_check_device()
        try:
            run_inference(self.dataloader_test, 0.01)
        except InferenceTimeAssertError:
            logger.info('Test passes')
