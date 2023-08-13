"""
All data manipulations functions such as building data loaders and performance check
"""
import logging
from datetime import datetime
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils import load_json, visualize_conf_matrix
logger = logging.getLogger('process_log')
result_sep = '\n'


def data_loader(config=load_json('configs.json')):
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        # transforms.RandomCrop(size=16),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
    return data_train, data_test, dataloader_train, dataloader_test


def performance_check(dataloader_test, dataset_test, model):
    predicted_labels, true_labels = [], []
    classes = model.common_attr['classes']
    result_str = f'{result_sep}Execution Timestamp: {datetime.now()}{result_sep}{result_sep}' \
                 f'| model name | num of images | class_name | accuracy |{result_sep}' \
                 f'| :---- | :---- | :---- | :---- |{result_sep}'
    correct, incorrect, total = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            # calculate outputs by running images through the model
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # prepare for performance statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend([classes[i] for i in labels.tolist()])
            predicted_labels.extend([classes[i] for i in predicted.tolist()])

    from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
    cm_overall_classes = confusion_matrix(true_labels, predicted_labels, labels=classes)
    visualize_conf_matrix(cm_overall_classes, classes, 'cm_overall_classes.png')

    logger.info(f'Accuracy of model {model.name} on the 10,000 test images: {100 * correct // total}%')
    result_str = result_str + f'|{model.name}|10,000|ALL|{100 * correct // total}%|{result_sep}'

    # prepare to count predictions for each class
    correct_pred = {class_name: 0 for class_name in classes}
    total_pred = {class_name: 0 for class_name in classes}
    
    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print and output accuracy for each class
    for class_name, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[class_name]
        logger.info(f'Accuracy for class: {class_name:5s} is {accuracy:.1f}%')
        result_str = result_str + f'|{model.name}|{total_pred[class_name]}|{class_name:5s}|{accuracy:.2f}%|{result_sep}'

    cm_by_classes = multilabel_confusion_matrix(true_labels, predicted_labels, labels=classes)
    # TODO enable or remove?
    for i, conf_matrix in enumerate(cm_by_classes):
        visualize_conf_matrix(conf_matrix, ['Positive', 'Negative'], f'cm_{classes[i]}.png')
    result_str = result_str + result_sep + str(model) + result_sep
    # export performance in table format
    result_file = model.common_attr['performance_file']
    # cm_file = model.common_attr['cm_file']
    cm_file = 'outputs/confusion_matrix.txt'
    with open(result_file, 'a') as file:
        file.write(result_str)
    # export confusion metrics to file
    np.savetxt(cm_file, cm_overall_classes, fmt='%d')
    # np.savetxt(cm_file, cm_by_classes)
    logger.info(f'Performance is exported to file {result_file}\nConfusion metrics are exported to file {cm_file}')
