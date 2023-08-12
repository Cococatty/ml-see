import logging
logger = logging.getLogger('process_log')
import torch


def performance_check(dataloader_test, model):
# def inference(dataloader_test, model):
    classes = model.common_attr['classes']
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            # calculate outputs by running images through the model
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    logger.info(f'Accuracy of the {model.name} on the 10000 test images: {100 * correct // total} %')
    
    # prepare to count predictions for each class
    correct_pred = {class_name: 0 for class_name in classes}
    total_pred = {class_name: 0 for class_name in classes}
    
    # again no gradients needed
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

    # print accuracy for each class
    for class_name, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[class_name]
        logger.info(f'Accuracy for class: {class_name:5s} is {accuracy:.1f} %')
