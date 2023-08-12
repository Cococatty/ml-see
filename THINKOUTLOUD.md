Hey there, Car! 
A Fun Car Model
Carina Zheng

# Reading the Technical Test Document
After scan-reading the entire document, the Requirements section lists out requirements that ALL 3 tasks must meet. Therefore, it's essential to ensure I understand all requirement items and plan my tasks with them in mind.

Simplifying with my own language, key Requirements for the final results:
1. the model should be completely brand new work; no pre-train or transfer-learning is allowed.
2. CPU enabled

Last but not least, how shall the thinking process be shared? Word - formatting takes too much time; and code may be involved, let's do LaTex!

Task
Design, train, and validate a model based on the CIFAR10 dataset (Canadian Institute For Advanced Research). You can view a getting started guide to downloading the dataset and training a simple CNN model using PyTorch here: 
[CIFAR10](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

- training data: cifar-10 training set
- validation and testing: cifar-10 test set

## LEVEL 1

- a public git repo
- A python script(s) which trains a model from scratch and saves it to a file
- A python script(s) which tests the generated model
- documentation:
    -- how to run both scripts
    -- A document that describes the chosen model and structure, and the result of the model, i.e. several measures of accuracy and performance

## LEVEL 2

- Optimization:
    - high accuracy 
    - a small model size
    - must contain less than 1 million parameters
- model performance analysis:
    - confusion matrix
    - breakdown accuracy for each label/class
- Measure the inference time of the model in milliseconds per image per CPU
- Measure the number of parameters of the model

Bonus: Measurements are computable dynamically when running the test script

## LEVEL 3
- Show your work: Document your thinking process, findings and experiments.
- Imagine if you had 3 months to work on this model: Describe areas of future research, techniques you might implement, experiments you might run, and specific areas where you believe the model could be improved.
- Reference any academic research that you have consulted when working on this test.

Requirements

- The classifier must be capable of running on a standard, consumer laptop, and be CPU driven.
- The inference will also run on the laptop, so size ( ≤ 1 million parameters), and inference time of the model is important ( ≤ 20ms inference time on a single CPU thread).
- You may use any preferred framework, e.g. PyTorch, TensorFlow, Keras, MXNet, Caffe, Theano, etc.
- Use of a pre-trained model or transfer learning is not allowed for this task; the solution must be entirely your own work.
- Use ChatGPT cautiously.
- Show thinking and how to plan professionally in: project, code, and other resources

## Instant Thoughts
- uint8 to start with
- simple to start with
- modulize the entire thing

## Available Resources
- got questions: Julian
- report writing: README Markdown
- code compiling: local dev env, with a newly created GitHub repo


# About the Data
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.

With the research of the photography domain, Commercial Photography is one of the highest paid Photography Jobs. \href{https://photographycourse.net/highest-paying-photography-jobs/}{17 Highest Paying Photography Jobs} 

therefore, automobile is chosen with the more future development, and future extension on trucks


# About the Model

# Questions
1. Is the model training all classes? Or only one or a few chosen classes?
2. Any specific goal of this project?


