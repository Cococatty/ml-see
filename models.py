import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# local modules
from utils import load_json
logger = logging.getLogger('process_log')


class CNN(nn.Module):
    def __init__(self, name):
        super().__init__()
        # add basic attributes
        self.name = name
        # load relevant configs from file
        self.hyperparams = load_json('configs.json')[name]
        self.common_attr = load_json('configs.json')['common_att']
        logger.info(f'{name} model is created')

    def save_model(self):
        os.makedirs(self.common_attr['output_path'], exist_ok=True)
        output_path = os.path.join(self.common_attr['output_path'], self.name+'.pth')
        torch.save(self.state_dict(), output_path)
        logger.info(f'{self.name} model is saved to file: {output_path}')


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
