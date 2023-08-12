import logging
import torch
import torch.nn as nn
logger = logging.getLogger('process_log')


def train_model(model, dataloader_train):
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=model.hyperparams['lr'], momentum=model.hyperparams['momentum'])

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
        # checkpoint
        if epoch % 2 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': running_loss
                        }, f'{model.name}_{epoch}.pt')
    logger.info(f'Finished training {model.name}')
    return model


def load_model(model, model_path, images):
    model.load_state_dict(torch.load(model_path))
