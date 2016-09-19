"""
network.py
~~~~~~~~~~

Feedforward neural network with stochastic gradient descent learning algorithm.  
Gradients are calculated using backpropagation.  

Naive implementation. 
"""

####
import json
import random
import numpy as np

####
class net(object):

	# [layers]: list with number of neurons for each layer. Ex: [3, 2, 1] defines a 3 layer network with 3/2/1 respective number of neurons
    def __init__(self, layers):
        # biases and weights are initialized randomly using a Gaussian with mean 0 and variance 1.  
        # first layer is an input layer, no biases will be assigned for these neurons.
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])]

    # [a]: input data
    def feedforward(self, a):
        # returns network output
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    # [a]: input data
    def predict(self, a):
        # returns network output
        predictions = []
        for p in a:
            result = self.feedforward(p)
            result = [int(round(v)) for v in result]
            predictions.append(result)

        return predictions

    # [training]: list of (x, y) input/output tuples for training
    # [epochs]: number of epochs to train
    # [batch_size]: number of random neurons selected for training each epoch
    # [eta]: learning rate
    # [test]: optional evaluation data. if provided, partial progress will be printed out
    def train(self, training, epochs, batch_size, eta, test = None):
    	# trains network using stochastic gradient descent.
        n = len(training)

        for j in xrange(epochs):
            random.shuffle(training)
            batches = [training[k: k + batch_size] for k in xrange(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)

            if test:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test), len(test))
            else:
                print "Epoch {0} complete".format(j)

    # [batch]: list of (x, y) tuples
    # [eta]: learning rate
    def update_batch(self, batch, eta):
        # updates weights and biases through gradient descent using backpropagation.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]

    # [x]: input data
    # [y]: output 
    def backprop(self, x, y):
        # returns (nabla_b, nabla_w) gradient tuple for the cost function 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        zs = [] 
        activation = x
        activations = [x]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = quadratic_delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # layer = 1 is last layer
        # layer = 2 is the second-last layer
        # ...
        for layer in xrange(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_delta(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())

        return (nabla_b, nabla_w)

    # [test]: list of (x, y) test data tuples
    def evaluate(self, test):
        # returns number of test inputs for which the neural network outputs correct results. 
        # neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
        results = [(self.feedforward(x), y) for (x, y) in test]
        results = [([int(round(v)) for v in x], [int(round(v)) for v in y]) for (x, y) in results]

        # count correct results
        return sum(int(x == y) for (x, y) in results)

    #
    def save(self, filename):
        # Save the neural network to disk
        data = {"layers": self.layers,
                "biases": [b.tolist() for b in self.biases],
                "weights": [w.tolist() for w in self.weights]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### 
def load(filename):
    # load neural network from disk
    # returns network instance
    file = open(filename, "r")
    data = json.load(file)
    file.close()
    
    net = net(data["layers"])
    net.biases = [np.array(b) for b in data["biases"]]
    net.weights = [np.array(w) for w in data["weights"]]

    return net

####
def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_delta(z):
    # sigmoid derivative
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
	# hyperbolic tangent function
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_delta(z):
	# hyperbolic tangent function
	return 1 - tanh(z) ** 2

def quadratic(a, y):
    # cost associated with output [a] and desired output [y]
    return 0.5 * np.linalg.norm(a - y) ** 2

def quadratic_delta(z, a, y):
    # error delta from output layer
    return (a - y) * sigmoid_delta(z)

def crossentropy(a, y):
    # cost associated with output [a] and desired output [y]
    return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

def crossentropy_delta(z, a, y):
    # error delta from output layer
    return (a - y)
