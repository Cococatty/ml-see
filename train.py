import logging
import os.path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import visualize_conf_matrix

logger = logging.getLogger('process_log')


def train_model(model, dataloader_train):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=model.hyperparams['lr'], momentum=model.hyperparams['momentum'])

    logger.info(f'Start training model {model.name}')

    for epoch in range(model.common_attr['epoch']):  # loop over the dataset multiple times
        logger.info(f'Training epoch {epoch}')
        running_loss = 0.0

        for i, data in enumerate(dataloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        logger.info(f'Finished training epoch {epoch}')
        # save checkpoint every 2nd epoch
        if epoch % 2 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': running_loss
                        }, f'{model.model_dir}_{epoch}.pt')
    logger.info(f'Finished training {model.name}')
    return model


def evaluate_model(dataloader_test, model):
    correct, incorrect, total = 0, 0, 0
    predicted_labels, true_labels = [], []
    classes = model.common_attr['classes']
    result_sep = model.common_attr['result_sep']
    result_str = f'{result_sep}Execution Timestamp: {datetime.now()}{result_sep}{result_sep}' \
                 f'| model name | num of images | class_name | accuracy |{result_sep}' \
                 f'| :---- | :---- | :---- | :---- |{result_sep}'
    model.eval()

    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            # calculate outputs by running images through the model
            outputs = model(images)
            # the class with the highest energy is chosen as final prediction
            _, predicted = torch.max(outputs.data, 1)
            # prepare for performance statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend([classes[i] for i in labels.tolist()])
            predicted_labels.extend([classes[i] for i in predicted.tolist()])

    from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
    cm_overall_classes = confusion_matrix(true_labels, predicted_labels, labels=classes)
    visualize_conf_matrix(cm_overall_classes, classes, f'{model.model_dir}/cm_overall_classes.png')

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

    # build confusion matrix for each class
    cm_by_classes = multilabel_confusion_matrix(true_labels, predicted_labels, labels=classes)
    for i, conf_matrix in enumerate(cm_by_classes):
        visualize_conf_matrix(conf_matrix, ['Positive', 'Negative'], f'{model.model_dir}/cm_{model.name}_{classes[i]}.png')

    # export performance results to file
    result_str = result_str + result_sep + str(model) + result_sep
    # export performance in table format
    result_file = model.common_attr['performance_file']
    cm_file = os.path.join(model.model_dir, model.common_attr['cm_file'])
    with open(result_file, 'a') as file:
        file.write(result_str)
    # export confusion metrics to file
    # TODO improvement: writing metrics to performance file
    np.savetxt(cm_file, cm_overall_classes, fmt='%d')
    logger.info(f'Performance is exported to file {result_file}\nConfusion metrics are exported to file {cm_file}')
