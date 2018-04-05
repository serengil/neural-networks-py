import math
import numpy as np
import time

#------------------------

x = np.array( #xor dataset
	[ #bias, #x1, #x2
		[[1],[0],[0]], #instance 1
		[[1],[0],[1]], #instance 2
		[[1],[1],[0]], #instance 3
		[[1],[1],[1]]  #instace 4
	]
)

y = np.array(
	[
		[0], #instance 1
		[1], #instance 2
		[1], #instance 3
		[0]  #instace 4
	]
)

#------------------------
epoch = 10000
hidden_layers = [3]

num_of_features = x.shape[1]-1 #minus 1 refers to bias
num_of_instances = x.shape[0]
num_of_layers = len(hidden_layers) + 2 #plus input layer and output layer

print("system configuration: epoch=",epoch,", hidden layers=",hidden_layers)

#------------------------
#weight initialization

w = [0 for i in range(num_of_layers-1)]

low = 0
high = 1

w[0] = np.random.uniform(low,high,(num_of_features + 1, hidden_layers[0])) #+1 refers to bias unit in input layer

if len(hidden_layers) > 1:
	for i in range(len(hidden_layers) - 1):
		w[i+1] = np.random.uniform(low,high,(hidden_layers[i] + 1, hidden_layers[i+1]))

w[num_of_layers-2] = np.random.uniform(low,high,(hidden_layers[len(hidden_layers) - 1] + 1, 1)) #+1 refers to bias unit in input layer, and 1 is number of nodes in output layer

#print("initial weights: ", w)

#------------------------

def sigmoid(netinput):
	netoutput = np.ones((netinput.shape[0] + 1, 1)) #ones because init values are same as bias unit. size of output is 1 more than input because of bias
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
		error = actual - predict
		
		sigmas = [i for i in range(num_of_layers)] #error should not be reflected to input layer
		
		sigmas[num_of_layers - 1] = error
		for j in range(num_of_layers - 2, -1, -1):
			
			if sigmas[j + 1].shape[0] == 1:
				sigmas[j] = w[j] * sigmas[j + 1]
			else:
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
	print(np.transpose(x[i][1:])," -> actual: ", actual,", predict: ", predict)
