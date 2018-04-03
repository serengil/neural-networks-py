import math
import numpy as np

#------------------------

x = np.array( #xor dataset
	[ #bias, #x1, #x2
		[[1],[0],[0]], #instance 1
		[[1],[0],[1]], #instance 2
		[[1],[1],[0]], #instance 3
		[[1],[1],[1]]  #instace 4
	]
)

w = np.array(
	[
		[ #weights for input layer to 1st hidden layer
			[0.8215133714710082, -4.781957888088778, 4.521206980948031],
			[-1.7254199547588138, -9.530462129807947, -8.932730568307496],
			[2.3874630239703, 9.221735768691351, 9.27410475328787]
		],
		[ #weights for hidden layer to output layer
			[3.233334754817538], 
			[-0.3269698166346504], 
			[6.817229313048568], 
			[-6.381026998906089]
		]
	]
)

c

num_of_layers = w.shape[0] + 1

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
for i in range(x.shape[0]):
	#print(x[i][1]," xor ", x[i][2], " = ", end='')
	nodes = applyFeedForward(x[i], w)
	print(nodes[num_of_layers - 1][1])