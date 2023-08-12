import logging
logger = logging.getLogger('process_log')
import torch
import torch.nn as nn


def train_model(model, dataloader_train):
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=model.hyperparams['lr'], momentum=model.hyperparams['momentum'])
    # criterion = nn.CrossEntropyLoss()

    logger.info(f'Start to train {model.name}')

    for epoch in range(1):  # loop over the dataset multiple times
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
        # TODO add breakpoint

    logger.info(f'Finished training {model.name}')
    return model


def load_model(model, model_path, images):
    model.load_state_dict(torch.load(model_path))
    outputs = model(images)

# net.to(device)
# inputs, labels = data[0].to(device), data[1].to(device)
# del dataiter
