import logging
import unittest
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
# project modules
from models import CNN

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


class SimpleCNN(CNN):
    def __init__(self):
        super().__init__(name='simpleCNN')
        # hidden layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestModels(unittest.TestCase):
    def test_givenASimpleModel_whenQueryTotalParams_thenReturnTheTotalParams(self):
        result = SimpleCNN().calc_total_num_params()
        self.assertEqual(result, 62006)

    def test_givenABigModel_whenTotalParamsOverOneMil_thenReturnErrorAndStop(self):
        with self.assertRaises(ValueError):
            OversizeCNN().calc_total_num_params()
