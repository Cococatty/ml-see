# Reading the Technical Test Document
After scan-reading the entire document, the Requirements section lists out requirements that ALL 3 tasks must meet. Therefore, it's essential to ensure I understand all requirement items and plan my tasks with them in mind.

Simplifying with my own language, key Requirements for the final results:
1. the model should be completely brand new work; no pre-train or transfer-learning is allowed.
2. target environment keywords: standard, consumer laptop, CPU. that is, limited spec 
3. parameters size ≤ 1 million => add validation from the beginning
4. model inference time ≤ 20ms on a single CPU thread => add validation from the beginning
5. Use ChatGPT cautiously
6. Show professional thinking and planning

Last but not least, how shall I share my thinking process?
- Word - formatting takes too much time, and code may be involved, hence, Word is not chosen
- LaTex - easy to format but require extra effort within limited timeframe
- Markdown - nice and simple, and can be stored along with the GitHub repo

## Break down the Tasks
### Understand the Data
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. -- detail from 
official CIFAR-10 website: https://www.cs.toronto.edu/~kriz/cifar.html

- training data: cifar-10 training set
- validation and testing: cifar-10 test set

### LEVEL 1

-[x] a public git repo
-[x] A python script(s) which trains a model from scratch and saves it to a file
-[x] A python script(s) which tests the generated model
-[x] documentation:
  - scripts running instructions 
  - describes the chosen model and structure, and the result of the model, i.e. several measures of accuracy and performance

### LEVEL 2

-[x] Optimization:
    -[x] try dropout layer
    -[x] high accuracy -- aim at 70%
    -[ ] consider use/convert to uint8 for models
    -[x] a small model size
    -[x] must contain less than 1 million parameters
-[x] model performance analysis:
    -[x] confusion matrix
    -[x] breakdown accuracy for each label/class
-[x] report inference time of the model in milliseconds per image per CPU
-[x] report number of parameters of the model
-[x] Bonus: Measurements are computable dynamically when running the test script

### LEVEL 3
- Show your work: Document your thinking process, findings and experiments.
-[x] Imagine if you had 3 months to work on this model: 
  -[x] Describe areas of future research, techniques you might implement, experiments you might run, 
  -[x] specific areas where you believe the model could be improved.
-[x] Reference any academic research that you have consulted when working on this test.

### Rough Delivery Plan
1. -[x] Determine and set up the environment and dependencies, Sat
2. -[x] Set up possibly required modules, Sat
3. -[x] Start documenting, Sat
4. -[x] First draft, Sat
5. -[x] Research and determine the model architecture, Sun
6. -[x] Design the model architecture, Sun
7. -[x] Build the desired model, Sun
8. -[x] Compile the model and ensure it works (level 1 completes), Sun
9. -[x] Optimization (level 2 starts), Sun
10. -[x] Detailed performance analysis, Sun
11. -[x] Finalize documentation, Mon
12. -[x] Clear all To-Do items, Mon
13. -[x] Clear all Checklist items, Mon
14. -[x] Submission, Mon 

### Available Resources
- got questions: Julian
- report writing: README, Markdown
- code compiling: local dev env, in a newly created GitHub repo

# About the Model
1. Start with something simple
2. Research academic paper

# Project To-Do
-[x] simple to start with
-[x] modularize the entire project
-[x] time the processes
-[x] add logger
-[x] add checkpoint
-[x] validation: parameters size ≤ 1 million
-[x] validation: inference time ≤ 20ms on a single CPU thread
-[x] export models performances to file
-[x] clear objects that are no longer required


# Completion Checklist
-[x] gh - README table of content
-[x] essential unit tests
-[x] remove unused dependencies
-[x] no comment-out code
-[x] code format
-[x] sufficient documentation, docstring, comments
-[x] clear all items in each level
-[x] is checkpoint still required for training lasts less than 5min)?


# Documentation
- PEP 257 – Docstring Conventions: https://peps.python.org/pep-0257/
