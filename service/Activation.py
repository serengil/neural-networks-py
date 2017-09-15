import math

class Activation:

    def activate(activation_function, x):

        f = 0

        if activation_function == 'sigmoid':

            f = 1 / (1 + math.exp(-1*x))
                    
        return f
    
    def derivative(activation_function, y, x):

        d = 0

        if activation_function == 'sigmoid':

            d = y * (1 - y)
        
        return d
