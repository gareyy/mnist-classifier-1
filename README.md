# MNIST-classifier-1

basic mnist classifier with tkinter based frontend

it doesn't really work well because data from user input can only have values of 0 or 1, whilst the dataset has datapoints with values between 0 and 1. I may need to round all the elements of datapoints in the dataset to 0 or 1.

Classifier model based on a neural network, with 4 hidden layers, each with a size of 512 parameters, each using ReLU. Trained using stochastic gradient descent with a really tiny but fixed learning rate. Stopping point for training happens when test dataset accuracy is greater than 95%.
