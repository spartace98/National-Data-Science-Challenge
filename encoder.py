import numpy as np
import pandas as pd

import random
import math

### Might wanna change to leaky ReLU
# Activation function (ReLU)
def relu(x):
    x[x<0] = 0
    return x

# Derivative of ReLU
def drelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

# Softmax function for output probabilities that sum to 1
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1)

# Sigmoid function for the fully connected net
def sigmoid(x):
    return np.tanh(x)

# Derivative of activation function (1.0 - y^2), in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.00 - np.square(sigmoid(y))
 
 # Mean squared cost function
def cost_fn(a, y):
    return 0.5 * np.mean(np.square(a - y))

#def cost_fn(a, y):
#    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def cost_delta(a, y):
    return (a - y)

class CNN:
    def __init__(self, number_of_convs, hidden_layer_size, number_of_categories):
        self.number_of_convs = number_of_convs

        # WEIGHT ARRAYS
        self.conv_weights = [np.random.randn(2, 1) for i in range(number_of_convs)]
        self.fullc_weights = [np.random.randn(number_of_convs, hidden_layer_size), np.random.randn(hidden_layer_size, number_of_categories)]

        # CACHES
        self.middle_conv_cache = []
        self.conv_cache = []
        self.input_layer_cache = []
        self.hidden_layer_cache = [0] * 2

        # VARIABLES
        self.learning_rate = 0.8


    # First function for input to pass through
    # Convolves over input using weights to produce set of convolutions
    # Passes set of convolutions to feedforward function
    def convolve(self, x):
        conv_layer_matrix = np.array([])
        for i in range(self.number_of_convs):
            newArray = np.array([])
            j = 0

            while(j < len(x) - 1):
                couples = np.array([x[j], x[j+1]])
                newArray = np.append(newArray, np.dot(np.transpose(couples), self.conv_weights[i]))
                j += 1
            conv_layer_matrix = np.append(conv_layer_matrix, newArray)
        conv_layer_matrix = np.reshape(conv_layer_matrix, (self.number_of_convs, len(x) - 1))
        conv_layer_matrix = relu(conv_layer_matrix)
        return self.feedforward(conv_layer_matrix)

    # Recursive function that stacks the convolutions 
    # Calls fullyconnected layer when the convulutions are 1 x 1
    def feedforward(self, conv_layer):
        if len(conv_layer[0]) == 1:
            self.conv_cache = conv_layer
            return self.fullyconnected(conv_layer)

        self.middle_conv_cache.append(conv_layer)

        next_conv_layer = np.array([])
        for i in range(len(conv_layer)):
            conv = conv_layer[i]

            newArray = np.array([])
            j = 0

            while(j < len(conv) - 1):
                couples = np.array([conv[j], conv[j+1]])
                newArray = np.append(newArray, np.dot(np.transpose(couples), self.conv_weights[i]))
                j += 1
            next_conv_layer = np.append(next_conv_layer, newArray)
        next_conv_layer = np.reshape(next_conv_layer, (self.number_of_convs, len(conv_layer[0]) - 1))
        next_conv_layer = relu(next_conv_layer)
        return self.feedforward(next_conv_layer)

    # Just 1 hidden layer
    # Output layer is 58 units
    # Returns the softmax (decimal closest to 1 is the answer)
    def fullyconnected(self, conv_layer):
        self.input_layer_cache = np.dot(np.transpose(conv_layer), self.fullc_weights[0])
        self.hidden_layer_cache[0] = sigmoid(self.input_layer_cache)
        self.hidden_layer_cache[1] = np.dot(self.hidden_layer_cache[0], self.fullc_weights[1])
        output_layer = sigmoid(self.hidden_layer_cache[1])
        return softmax(output_layer)


    # Given a certain input, checks if the output of the NN matches the expected output
    def check(self, input, expected_output):
        output = self.convolve(input)
        
        if np.argmax(output) == np.argmax(expected_output):
            return 1
        else:
            return 0

    def descend(self, input, expected_output):
        output = self.convolve(input)


        dLdO = cost_delta(output, expected_output)

        # Backprop through Softmax and Sigmoid
        # Gets delta for last weight matrix
        dOdW2 = np.dot(np.transpose(np.multiply(output, np.subtract(-1.0 * output, 1.0))), self.hidden_layer_cache[0])
        dLdW2 = np.dot(np.transpose(np.dot(dLdO, dOdW2)), dsigmoid(self.hidden_layer_cache[1]))
        
        # Backprop through sigmoid
        # Gets delta fdr second last weight matrix
        dOdH = np.dot(dLdO, dOdW2)
        dLdW1 = np.dot(self.conv_cache, np.multiply(dOdH, dsigmoid(self.input_layer_cache)))



        # Backprop though CNN
        dLdH = np.dot(dLdW2, np.transpose(dLdO))
        dLdC = np.dot(np.multiply(self.fullc_weights[0], dsigmoid(self.input_layer_cache)), dLdH)
        dC = np.multiply(dLdC, drelu(self.conv_cache))

        dFs = [np.zeros((2, 1)) for i in self.conv_weights] 
        # Iterates through the convolution levels vertically from the bottom
        for h in range(self.number_of_convs):
            for i, level in reversed(list(enumerate(self.middle_conv_cache))):
                if i == 0:
                    break
                thislevel = level[h]
                levelabove = self.middle_conv_cache[i-1][h]

                dFs[h][0] += np.sum(np.multiply(dC[h], thislevel[:-1]))
                dFs[h][1] += np.sum(np.multiply(dC[h], thislevel[1:]))

                break

        # Update convolution weights
        self.conv_weights = np.subtract(self.conv_weights, dFs)

        # Update the fully connected weights by the deltas
        self.fullc_weights[1] -= self.learning_rate * dLdW2
        self.fullc_weights[0] -= self.learning_rate * dLdW1

        print(cost_fn(output, expected_output), "\r", end='')
