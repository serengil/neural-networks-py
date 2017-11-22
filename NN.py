from entity.Node import Node
from service.NetworkStructure import NetworkStructure
from service.NetworkConnection import NetworkConnection
from service.NetworkLearning import NetworkLearning
from service.FeatureNormalization import FeatureNormalization
import matplotlib.pyplot as plt

#--------------------------------
#historical data
instances = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] #x1, x2, result
#--------------------------------
dump = True
epoch = 10000
activation_function = 'sigmoid'
learning_rate = 0.1

#tuning parameters
applyAdaptiveLearning = False
momentum = 0
#--------------------------------

instances = FeatureNormalization.normalize(instances, activation_function)
print("normalized dataset: ", instances)

#--------------------------------

num_of_features = len(instances[0]) - 1 #an instance consisting of features and output. that's why minus 1
hidden_nodes = [3] #use variable as [3, 3] to have 2 hidden layers consisting of 3 units in each layer

#--------------------------------

nodes = NetworkStructure.create_nodes(dump, num_of_features, hidden_nodes)

weights = NetworkConnection.create_weights(dump, nodes, num_of_features, hidden_nodes)

previous_cost = 0 #applying adaptive learning rate requires to store previous cost

cost_history = []

for i in range(epoch+1):
    
    nodes, weights = NetworkLearning.applyBackpropagation(dump, instances, nodes, weights, activation_function, learning_rate, momentum)
    J = NetworkLearning.calculate_cost(dump, instances, nodes, weights, activation_function)
    
    if i % 100 == 0:
        print(i,".\t",J)

        if i > 0:
            cost_history.append(J)

    #-----------------------------------

    if applyAdaptiveLearning == True:
        if J < previous_cost:
            learning_rate = learning_rate + 0.1
        else:
            learning_rate = learning_rate - 0.5*learning_rate
        
        previous_cost = J * 1
    #end of adaptive learning block

    #-----------------------------------

plt.plot(cost_history)
plt.show()

for i in range(len(instances)):
    instance = instances[i]
    NetworkLearning.applyForwardPropagation(dump, nodes, weights, instance, activation_function)
    print("actual: ",instance[len(instance)-1]," - prediction: ",nodes[len(nodes)-1].get_net_value())
