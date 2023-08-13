import logging
logger = logging.getLogger('process_log')


def set_check_device():
    import torch
    device = torch.device('cpu')
    logger.info(f'device is set to {device}')


def load_json(file_path='configs.json'):
    import json
    with open(file_path) as f:
        json_data = json.load(f)
    return json_data


def visualize_conf_matrix(cm, classes, file_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(13,8))
    sns.heatmap(cm, annot=True, fmt='d').set(xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prediction', size=13)
    plt.ylabel('Truth',  size=13)
    plt.savefig(f'outputs/{file_name}')
    # plt.imsave(f'outputs/{file_name}')
