# neural-networks-py
This project includes core neural networks implementation with python (3.5)

Run NN.py in the root directory.

# Configuration

Load historical data in the instances variable in the main file. For instance, the following variable states Exclusive OR (XOR) dataset. Last item of an instance states results whereas other items state input features.

instances = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] #x1, x2, result

In other words, 4th item of instances, [1, 1, 0], means x1 = 1, x2 = 1 and result = 0

Moreover, you should tune the following variables for different datasets.

dump = True
epoch = 10000
activation_function = 'sigmoid' #now, it works for sigmoid only.
learning_rate = 0.1

#tuning parameters
applyAdaptiveLearning = False
momentum = 0

PS: java implementation of this project is already shared in the following link
https://github.com/serengil/neural-networks
