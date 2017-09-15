from entity.Node import Node

class NetworkStructure:

    def create_nodes(dump, num_of_features, hidden_nodes):

        nodes = []

        #--------------------------------
        #input layer

        if dump == True:
            print("input layer: ", end='')

        #--------------------------------
        #bias unit

        nodeIndex = 0

        node = Node()
        node.set_level(0)
        node.set_item(0)
        node.set_label("+1")
        node.set_layer_name("input layer")
        node.set_index(nodeIndex)
        node.set_is_bias_unit(True)

        nodes.append(node)

        if dump == True:
            print(node.get_label(), "\t", end='')

        nodeIndex = nodeIndex + 1

        #--------------------------------

        for i in range(num_of_features):
            
            if dump == True:
                print("V"+str(i+1)+"\t", end='')
            
            node = Node()
            node.set_level(0)
            node.set_item(i+1)
            node.set_label(("V"+str(i+1)))
            node.set_layer_name("input layer")
            node.set_index(nodeIndex)
            node.set_is_bias_unit(False)

            nodes.append(node)

            nodeIndex = nodeIndex + 1

        print("")

        ##input layer end

        #--------------------------------
        #hidden layers

        for i in range(len(hidden_nodes)):
            
            if dump == True:
                #print("hidden layer #",i, ": ", end='')
                print("hidden layer: ", end='')
            
            #bias unit
            node = Node()
            node.set_level(i+1)
            node.set_item(0)
            node.set_label("+1")
            node.set_layer_name(("hidden layer (",i+1,")"))
            node.set_index(nodeIndex)
            node.set_is_bias_unit(True)

            nodes.append(node)

            if dump == True:
                print(node.get_label(),"\t", end='')

            nodeIndex = nodeIndex + 1
            
            for j in range(hidden_nodes[i]):
                node = Node()
                node.set_level(i+1)
                node.set_item(j+1)
                node.set_label(("N["+str(i+1)+"]["+str(j+1)+"]"))
                node.set_layer_name(("hidden layer (",i+1,")"))
                node.set_index(nodeIndex)
                node.set_is_bias_unit(False)

                nodes.append(node)

                if dump == True:
                    print(node.get_label(),"\t", end='')

                nodeIndex = nodeIndex + 1
            
            if dump == True:
                print("")

        #--------------------------------

        #output layer

        node = Node()
        node.set_level(1+len(hidden_nodes))
        node.set_item(1)
        node.set_label("output")
        node.set_layer_name(("output layer"))
        node.set_index(nodeIndex)
        node.set_is_bias_unit(False)

        if dump == True:
            print("output layer: output")

        nodes.append(node)

        return nodes