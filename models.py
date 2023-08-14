import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# local modules
from utils import load_json
from exceptions import ParamSizeAssertError
logger = logging.getLogger('process_log')


class CNN(nn.Module):
    def __init__(self, name):
        super().__init__()
        # add basic attributes
        self.name = name
        self.files_attr = load_json('configs.json')['files_attr']
        self.hyperparams = load_json('configs.json')[name]
        self.classes = load_json('configs.json')['classes']
        self.n_epochs = load_json('configs.json')['n_epochs']
        self.model_dir = os.path.join(self.files_attr['output_path'], self.name)
        os.makedirs(self.model_dir, exist_ok=True)
        # load relevant configs from file
        logger.info(f'Model {name} is created')

    def calc_total_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        if total_params > 1000000:
            logger.error(f'Model {self.name} parameters size ({total_params}) > 1 million')
            raise ParamSizeAssertError(f'Model {self.name} parameters size ({total_params}) > 1 million')
        elif total_params < 10:
            logger.error(f'Model {self.name} has unusually small number of parameters: {total_params}')
            raise ParamSizeAssertError(f'Model {self.name} has unusually small number of parameters')
        else:
            logger.info(f'Model {self.name} parameters size is {total_params}')
        return total_params


def save_model(model, output_path=None):
    if output_path is None:
        output_path = model.model_dir
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, model.name+'.pth')
    torch.save(model, model_path)
    logger.info(f'{model.name} model is saved to {model_path}')


class SimpleCNN(CNN):
    """The current best-performance model"""
    def __init__(self):
        super().__init__(name='SimpleCNN')
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool_max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.4)
        # fully connected layers        
        self.fc1 = nn.Linear(in_features=32*8*8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.num_params = self.calc_total_num_params()

    def forward(self, x):
        # x = self.pool_max(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.pool_max(x)
        x = F.relu(self.conv2(x))
        x = self.pool_max(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class TestCNN(CNN):
    def __init__(self):
        super().__init__(name='TestCNN')
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool_max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.4)
        # fully connected layers        
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.num_params = self.calc_total_num_params()        

    def forward(self, x):
        x = self.pool_max(F.relu(self.conv1(x)))
        x = self.pool_max(F.relu(self.conv2(x)))
        x = self.pool_max(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
