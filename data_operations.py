"""
All data manipulations functions such as building data loaders and performance check
"""
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from utils import load_json

logger = logging.getLogger('process_log')


class GrayscaleTransform(object):
    def __call__(self, img):
        return img.convert('L')


def data_loader(config=load_json('configs.json')):
    transform = transforms.Compose([
        GrayscaleTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   batch_size=config['dataloader_train']['batch_size'],
                                                   shuffle=config['dataloader_train']['shuffle'],
                                                   num_workers=config['dataloader_train']['num_workers'])

    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=config['dataloader_test']['batch_size'],
                                                  shuffle=config['dataloader_test']['shuffle'],
                                                  num_workers=config['dataloader_test']['num_workers'])

    logger.info('Data loaders for train and test are created')
    del data_train, data_test, transform
    return dataloader_train, dataloader_test
