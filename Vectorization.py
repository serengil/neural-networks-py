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

#random initial weights
w = np.array(
	[
		[ #weights for input layer to 1st hidden layer
			[0.441461337833972, -0.9385618776407734, 0.6994894097020672],
			[0.10683441180281483, -0.9344176790118245, 0.4717265671227009],
			[0.1624858934305029, -0.9314202266783156, -0.3284522527544532]
		],
		[ #weights for hidden layer to output layer
			[0.26537295294388175], 
			[-0.17541706129386947], 
			[0.7877122949375102], 
			[-0.8067422017211016]
		]
	]
)

num_of_layers = w.shape[0] + 1
num_of_features = x.shape[1] - 1 #minus bias unit
num_of_instances = x.shape[0]

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
	
	return netoutput

#------------------------

start_time = time.time()

for epoch in range(10000):
	for i in range(num_of_instances):
		instance = x[i]
		nodes = applyFeedForward(instance, w)
		
		predict = nodes[num_of_layers - 1][1]
		actual = y[i]
		error = actual - predict
		
		
		sigmas = [i for i in range(num_of_layers)] #error should not be reflected to input layer
		
		sigmas[num_of_layers - 1] = error
		for j in range(num_of_layers - 2, -1, -1):
			
			if sigmas[j + 1].shape[0] == 1:
				sigmas[j] = w[j] * sigmas[j + 1]
			else:
				sigmas[j] = np.matmul(np.transpose(w[j]), sigmas[j + 1][1:])
			
		#----------------------------------
		
		derivative_of_sigmoid = nodes * (np.array([1]) - nodes) #element wise multiplication and scalar multiplication
		sigmas = derivative_of_sigmoid * sigmas
		
		for j in range(num_of_layers - 1):
			delta = nodes[j] * np.transpose(sigmas[j+1][1:])
			w[j] = w[j] + np.array([0.1]) * delta
	
#training end
#---------------------
print("--- execution for vectorization lasts %s seconds ---" % (time.time() - start_time))

for i in range(num_of_instances):
	nodes = applyFeedForward(x[i], w)
	predict = nodes[num_of_layers - 1][1]
	actual = y[i][0]
	print(np.transpose(x[i][1:])," -> actual: ", actual,", predict: ", predict)
