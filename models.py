import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# local modules
from utils import load_json
from exceptions import ParamSizeValueError
logger = logging.getLogger('process_log')


class CNN(nn.Module):
    def __init__(self, name):
        super().__init__()
        # add basic attributes
        self.name = name
        # load relevant configs from file
        self.common_attr = load_json('configs.json')['common_att']
        self.hyperparams = load_json('configs.json')[name]
        self.num_params = self.calc_total_num_params()
        logger.info(f'{name} model is created')

    def calc_total_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        if total_params > 1000000:
            logger.error(f'Model {self.name} parameters size ({total_params}) > 1 million')
            raise ParamSizeValueError(f'Model {self.name} parameters size ({total_params}) > 1 million')
        else:
            logger.info(f'Model {self.name} parameters size is {total_params}')
        return total_params


def save_model(model, output_path=None):
    if output_path is None:
        output_path = model.common_attr['output_path']
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, model.name+'.pth')
    torch.save(model, model_path)
    logger.info(f'{model.name} model is saved to {model_path}')


class SimpleCNN(CNN):
    def __init__(self):
        super().__init__(name='simpleCNN')
        # hidden layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.AvgPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # dropout layers
        # self.dropout = nn.Dropout(p=0.2)
        # self.dropout2 = nn.Dropout(p=0.2)
        # fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
