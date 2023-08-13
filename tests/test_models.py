import logging
import unittest
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
# project modules
from models import CNN, SimpleCNN, save_model
from exceptions import ParamSizeValueError

FILE = Path(__file__).resolve()
test_dir = FILE.parents[0]
project_dir = FILE.parents[1]
sys.path.append(str(project_dir))
sys.path.append(str(test_dir))
os.chdir(test_dir)
logger = logging.getLogger('unittest_log')


class OversizeCNN(CNN):
    def __init__(self):
        super().__init__(name='oversizeCNN')
        # hidden layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(16 * 25 * 25, 480)
        self.fc2 = nn.Linear(120, 168)
        self.fc3 = nn.Linear(84, 200)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# given a condition, when action, then result
class TestModels(unittest.TestCase):
    def setUp(self):
        self.simple_cnn = SimpleCNN()
        self.output_path = os.path.join(test_dir, 'outputs')
        self.cnn_model_path = os.path.join(self.output_path, 'simpleCNN.pth')
        self.cnn_state_path = os.path.join(self.output_path, 'simpleCNN_state.pth')

    @staticmethod
    def remove_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def test_givenASimpleModel_whenQueryTotalParams_thenReturnTheTotalParams(self):
        result = self.simple_cnn.calc_total_num_params()
        self.assertEqual(result, 62006)

    def test_givenABigModel_whenTotalParamsOverOneMil_thenReturnErrorAndStop(self):
        with self.assertRaises(ParamSizeValueError):
            OversizeCNN().calc_total_num_params()

    def test_givenAModel_whenModelExportIsRequired_thenSaveTheEntireModelAndStateDictToFiles(self):
        self.remove_file(self.cnn_model_path)
        save_model(self.simple_cnn, self.output_path)
        self.assertTrue(os.path.exists(self.cnn_model_path))

    def test_givenAModelFile_whenLoadIsRequired_thenModelIsLoadedFromFile(self):
        # TODO fix
        model = torch.load(self.cnn_model_path)
        # Compare the state dictionaries of the two models
        state_dict1 = self.simple_cnn.state_dict()
        state_dict2 = model.state_dict()

        # Assert that the state dictionaries are the same
        assert state_dict1.keys() == state_dict2.keys()
        for key in state_dict1.keys():
            assert torch.all(state_dict1[key] == state_dict2[key])
