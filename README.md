# About this Repository
- train.py:  trains the target model from scratch and saves it to a file
- test.py: tests the generated model
- .py: tests the generated model

# About the Model and Structure

# The Model Result
- i.e. several measures of accuracy and performance

# How to Setup the Running Environment
## Pre-requisite/Development Environment
Python 3.10.6

- Python3
- Pip3

# Set up the running environment

```
# verify local python3 version
python3 --version

# create the virtual environment
python3 -m venv venv_see
# activate the virtual environment
source venv_see/bin/activate
# install dependencies
pip3 install -r requirements.txt
```

# How to Run

```
# activate the virtual environment
source venv_see/bin/activate
python3 main.py
```
OR
```
venv_see/bin/python3 main.py
```
## Compiling Outputs
### Training and Evaluation
- log files
	- log_compilation.log: generated from training and evaluating the chosen model
	- log_inference.log: generated from running the inference against the chosen model
- in outputs directory:
	- models_performances_tables.txt: this text file contains performance raw outputs, such as accuracy of each class, model architecture used at the time (given no other versioning tools are used in this project), configurations and hyperparameters used
	- a new directory named by the chosen model, which contains checkpoint files, saved model file, overall confusion matrix and confusion metrics by classes


### Inference

# Models Performances Report

## Dataset Details
- Dataset: CIFAR-10
- Number of Classes: 10
- Total Number of Images: 60,000
- Training Samples: 50,000
- Test Samples: 10,000

## Model Architecture

- Model: SimpleCNN
- Architecture: A simple CNN model that is developed locally on a Dell Inspiron 15 (2015) with CPU.
- Number of Parameters: 268,362

Models performances comparison fetch from https://www.researchgate.net/figure/Ball-chart-reporting-the-Top-1-and-Top-5-accuracy-vs-computational-complexity-Top-1-and_fig1_328509150

![Models performances comparison](doc_imgs/Ball-chart-reporting-the-Top-1-and-Top-5-accuracy-vs-computational-complexity-Top-1-and.png)

ResNet50 architecture, fetched from https://www.researchgate.net/figure/The-framework-of-the-Resnet50-The-Resnet50-model-trained-on-ImageNet-which-is_fig3_344190091  

![ResNet50 architecture](doc_imgs/The-framework-of-the-Resnet50-The-Resnet50-model-trained-on-ImageNet-which-is.png)

YOLO architecture, fetch from https://www.researchgate.net/figure/YOLO-network-architecture-adapted-from-44_fig1_330484322

![YOLO architecture](doc_imgs/YOLO-network-architecture-adapted-from-44.png)

VGG16 architecture, fetch from https://neurohive.io/en/popular-networks/vgg16/

![VGG16 architecture](doc_imgs/vgg16.png)

[//]: # (https://deci.ai/blog/close-gap-cpu-performance-gpu-deep-learning-models/)

## Training Details

- Training Duration: 2250.66 seconds/37.5 minutes (average on 8 runs) on Dell, 240.53 seconds/4 minutes (average on 3 runs) on a borrowed MacBook Pro with M1 chip
- Optimizer: SGD - for its computational efficiency
- Learning Rate: 0.001
- Batch Size: [Batch size]
- Loss Function: CrossEntropyLoss
- Training Accuracy: [Training accuracy]

## Performance Metrics

- Test Accuracy: [Test accuracy]
- Confusion Matrix: [Insert confusion matrix if applicable]


## Inference Time

- Average Inference Time: [Average inference time per image]
- Hardware Used: [Description of hardware used]
