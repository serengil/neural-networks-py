import math
import numpy as np
import time

#------------------------
#system configuration

num_of_classes = 2 #1 for regression, n for classification
epoch = 10000
hidden_layers = [5, 5]

print("system configuration: epoch=",epoch,", hidden layers=",hidden_layers)

#------------------------
#trainset

x = np.array( #xor dataset
	[ #bias, #x1, #x2
		[[1],[0],[0]], #instance 1
		[[1],[0],[1]], #instance 2
		[[1],[1],[0]], #instance 3
		[[1],[1],[1]]  #instace 4
	]
)

if num_of_classes > 1: #classification
	y = np.array(
		[
			[[1],[0]],
			[[0],[1]],
			[[0],[1]],
			[[1],[0]],
		]
	)
else: #regression
	y = np.array(
		[
			[0], #instance 1
			[1], #instance 2
			[1], #instance 3
			[0]  #instace 4
		]
	)

"""
#suppose that traindata stored in this file. each line refers to an instance.
#final item of an instance is label (y) whereas the other ones are features  (x)
#now, loading trainset from external resource is available for regression problems
from numpy import genfromtxt
dataset = genfromtxt('../xor.csv', delimiter=',')
features = dataset[:,0:dataset.shape[1]-1]
labels = dataset[:,dataset.shape[1]-1:]

bias = np.array([1]);

x = [0 for i in range(features.shape[0])]
for i in range(features.shape[0]):
	x[i] = np.array([np.append([1], features[i])]).T

y = [0 for i in range(labels.shape[0])]
for i in range(labels.shape[0]):
	if num_of_classes == 1: #regression
		y[i] = np.array([labels[i]]).T
	else:
		encoded = [0 for i in range(num_of_classes)]
		encoded[int(labels[i])] = 1
		y[i] = np.array([encoded]).T

x = np.array(x)
y = np.array(y)
"""
#print("features: ",x)
#print("target: ",y)
#------------------------

num_of_features = x.shape[1]-1 #minus 1 refers to bias
num_of_instances = x.shape[0]
num_of_layers = len(hidden_layers) + 2 #plus input layer and output layer

#------------------------
def initialize_weights(rows, columns):
	weights = np.random.randn(rows+1, columns) #standard normal distribution, +1 refers to bias unit
	weights = weights * np.sqrt(2/rows)
	#weights = weights * np.sqrt(1/rows) #xavier initializtion
	#weights = weights * np.sqrt(6/(rows + columns)) #normalized initializtion
	return weights

#weight initialization
w = [0 for i in range(num_of_layers-1)]

w[0] = initialize_weights(0, num_of_features, hidden_layers[0])

if len(hidden_layers) > 1:
	for i in range(len(hidden_layers) - 1):
		w[i+1] = initialize_weights(i+1, hidden_layers[i], hidden_layers[i+1])

w[num_of_layers-2] = initialize_weights(num_of_layers-2, hidden_layers[len(hidden_layers) - 1], num_of_classes)

print("initial weights: ", w)
#------------------------

def sigmoid(netinput):
	netoutput = np.ones((netinput.shape[0] + 1, 1)) 
	#ones because init values are same as bias unit
	#size of output is 1 plus input because of bias
	
	for i in range(netinput.shape[0]):
		netoutput[i+1] = 1/(1 + math.exp(-netinput[i][0]))
	return netoutput

def applyFeedForward(x, w):

	netoutput = [i for i in range(num_of_layers)]
	netinput = [i for i in range(num_of_layers)]

	netoutput[0] = x
	for i in range(num_of_layers - 1):
		netinput[i+1] = np.matmul(np.transpose(w[i]), netoutput[i])
		netoutput[i+1] = sigmoid(netinput[i+1])	
	
	netoutput[num_of_layers-1] = netoutput[num_of_layers-1][1:] #bias should not exist in output
	return netoutput

#------------------------

start_time = time.time()

for epoch in range(epoch):
	for i in range(num_of_instances):
		instance = x[i]
		nodes = applyFeedForward(instance, w)
		
		predict = nodes[num_of_layers - 1]
		actual = y[i]
		#print("predict: ",predict,", actual: ",actual)
		error = actual - predict
		#print("error: ",error)
		
		sigmas = [i for i in range(num_of_layers)] #error should not be reflected to input layer
		
		sigmas[num_of_layers - 1] = error
		for j in range(num_of_layers - 2, -1, -1):
			
			if sigmas[j + 1].shape[0] == 1:
				sigmas[j] = w[j] * sigmas[j + 1]
			else:
				if j == num_of_layers - 2: #output layer has no bias unit
					sigmas[j] = np.matmul(w[j], sigmas[j + 1])
				else: #otherwise remove bias unit from the following node because it is not connected from previous layer
					sigmas[j] = np.matmul(w[j], sigmas[j + 1][1:])
			
		#----------------------------------
		
		derivative_of_sigmoid = nodes * (np.array([1]) - nodes) #element wise multiplication and scalar multiplication
		sigmas = derivative_of_sigmoid * sigmas
		
		for j in range(num_of_layers - 1):
			
			if j == num_of_layers - 2: #outputlayer
				delta = nodes[j] * np.transpose(sigmas[j+1]) #no bias exist in output layer
			else:
				delta = nodes[j] * np.transpose(sigmas[j+1][1:])

			w[j] = w[j] + np.array([0.1]) * delta
	
#training end
#---------------------
print("--- execution for vectorization lasts %s seconds ---" % (time.time() - start_time))

#print("final weights: ",w)

for i in range(num_of_instances):
	nodes = applyFeedForward(x[i], w)
	predict = nodes[num_of_layers - 1]
	actual = y[i]
	print(np.transpose(x[i][1:])," -> actual: ", np.transpose(actual)[0],", predict: ", np.transpose(predict)[0])
