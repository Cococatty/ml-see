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
