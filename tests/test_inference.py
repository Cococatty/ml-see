import logging
import unittest
import os
import sys
import threading
from pathlib import Path
# project modules
from utils import set_check_device
from models import SimpleCNN
from data_operations import data_loader
from inference import single_threaded_inference, run_inference

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

    @staticmethod
    def create_large_image():
        import numpy as np
        height, width, channels = 1000, 1500, 3
        large_image = np.zeros((height, width, channels), dtype=np.uint8)
        large_image[100:200, 200:400, 0] = 255
        return large_image

    def test_givenASimpleModel_whenInferenceRequired_thenLoadTheModel(self):
        dataloader_train, dataloader_test = data_loader()
        # cnn = SimpleCNN().load_state_dict('outputs/simpleCNN.pth')
        # thread = threading.Thread(target=single_threaded_inference, args=(cnn, dataloader_test))
        # thread.start()
        # thread.join()
        # TODO enable
        # self.assertEqual(result, 62006)

    def test_givenALargeImage_whenRunInferenceAndExceedTimeReq_thenGiveAWarning(self):
        set_check_device()
        _, dataloader_test = data_loader()
        self.withWar
        t_inf_start = process_time()
        run_inference(dataloader_test)
