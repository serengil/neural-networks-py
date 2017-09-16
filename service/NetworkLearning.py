from service.Activation import Activation

class NetworkLearning:

    def applyForwardPropagation(dump, nodes, weights, instance, activation_function):

        #transfer bias unit values as +1

        for j in range(len(nodes)):
            if nodes[j].get_is_bias_unit() == True:
                nodes[j].set_net_value(1)

        #------------------------------
        #tranfer instace features to input layer. activation function would not be applied for input layer.
        for j in range(len(instance) - 1): #final item is output of an instance, that's why len(instance) - 1 used to iterate on features

            var = instance[j]

            for k in range(len(nodes)):

                if j+1 == nodes[k].get_index():
                    nodes[k].set_net_value(var)
                    break
        
        #------------------------------

        for j in range(len(nodes)):

            if nodes[j].get_level() > 0 and nodes[j].get_is_bias_unit() == False:

                net_input = 0
                net_output = 0

                target_index = nodes[j].get_index()

                for k in range(len(weights)):

                    if target_index == weights[k].get_to_index():

                        wi = weights[k].get_value()

                        source_index = weights[k].get_from_index()

                        for m in range(len(nodes)):

                            if source_index == nodes[m].get_index():

                                xi = nodes[m].get_net_value()

                                net_input = net_input + (xi * wi)

                                #print(xi," * ", wi," + ", end='')

                                break
                
                #iterate on weights end
                
                net_output = Activation.activate(activation_function, net_input)
                nodes[j].set_net_input_value(net_input)
                nodes[j].set_net_value(net_output)
                
        #------------------------------
        
        return nodes

    def applyBackpropagation(dump, instances, nodes, weights, activation_function, learning_rate, momentum):

        num_of_features = len(instances[0]) - 1

        for i in range(len(instances)):

            #apply forward propagation first
            nodes = NetworkLearning.applyForwardPropagation(dump, nodes, weights, instances[i], activation_function)

            actual_value = instances[i][len(instances[0])-1]
            predicted_value = nodes[len(nodes) - 1].get_net_value()

            #print("actual: ",actual_value," - predicted:",predicted_value)

            small_delta = actual_value - predicted_value

            nodes[len(nodes) - 1].set_small_delta(small_delta)
            
            for j in range(len(nodes)-2, num_of_features, -1): #output delta is already calculated on the step above, that's why len(nodes)-2

                #look for connections including from nodes[j]

                target_index = nodes[j].get_index()

                sum_small_delta = 0

                for k in range(len(weights)):

                    if weights[k].get_from_index() == target_index:

                        affecting_theta = weights[k].get_value()
                        affetcting_small_delta = 1

                        target_small_delta_index = weights[k].get_to_index()

                        for m in range(len(nodes)):

                            if nodes[m].get_index() == target_small_delta_index:

                                affetcting_small_delta = nodes[m].get_small_delta()

                                break

                        #-------------------------

                        newly_small_delta = affecting_theta * affetcting_small_delta

                        sum_small_delta = sum_small_delta + newly_small_delta

                #---------------------------

                nodes[j].set_small_delta(sum_small_delta)

            #calculation of small deltas end

            #-------------------------------

            #apply stockastic gradient descent to update weights

            previous_derivative = 0 #applying momentum requires to store previous derivative
            
            for j in range(len(weights)):

                weight_from_node_value = 0
                weight_to_node_delta = 0
                weight_to_node_value = 0
                weight_to_node_net_input = 0

                for k in range(len(nodes)):

                    if nodes[k].get_index() == weights[j].get_from_index():

                        weight_from_node_value = nodes[k].get_net_value()

                    if nodes[k].get_index() == weights[j].get_to_index():

                        weight_to_node_delta = nodes[k].get_small_delta()
                        weight_to_node_value = nodes[k].get_net_value()
                        weight_to_node_net_input = nodes[k].get_net_input_value()

                #---------------------------
            
                derivative = weight_to_node_delta * Activation.derivative(activation_function, weight_to_node_value, weight_to_node_net_input) * weight_from_node_value

                weights[j].set_value(weights[j].get_value() + learning_rate * derivative + momentum * previous_derivative)

        return nodes, weights

    def calculate_cost(dump, instances, nodes, weights, activation_function):

        J = 0

        for i in range(len(instances)):

            instance = instances[i]

            nodes = NetworkLearning.applyForwardPropagation(dump, nodes, weights, instance, activation_function)

            predict = nodes[len(nodes)-1].get_net_value()
            actual = instances[i][len(instances[i])-1]

            #print("((",predict,"-",actual,")^2)/2 = ", end='')

            cost = (predict-actual)*(predict-actual)
            cost = cost / 2

            #print(cost)

            J = J + cost

        J = J / len(instances)

        return J
