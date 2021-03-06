"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):
    # constructor
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        # sizes is an array where each element stores the number of neurons per layer. take its length.
        self.num_layers = len(sizes)
        self.sizes = sizes
        # for every layer but the first (the ones containing perceptrons), create a y by 1 matrix
        # containing random values
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # create a y by x matrix, where y is the # neurons of every layer but the last, and x is the
        # no. of neurons of every layer but the first
        # i.e. list [2,3,1]
        # y = 2, then 3
        # x = 3, then 1
        # so weights is an array of matrices (first of size 2 by 3, then second of size 3 by 1)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # a = the input layer. Compute the output layer by formula (25)
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        # zip creates a arr of tuples
        # b = the ith biases (a y by 1 matrix)
        # w = the ith weights (a y by x matrix)
        for b, w in zip(self.biases, self.weights):
            # (25) a^(l) = sigmoid(w^(l) a^(l-1) + b^(l))
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        # if test_data exists, get the number of tests
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # original implementation used xrange, not supported by Python 3
        # j is
        for j in range(epochs):
            # mix up training data (a list of tuples (x,y) -> x = training input, y = desired output)
            random.shuffle(training_data)
            # divide up the training data into a list of batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                # range(start, stop, step)
                # n = # of training data
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # Update the network's weights and biases by applying gradient descent (1 iteration) using
                # backpropagation to a single mini batch.
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                # print the number of correct interpretations, like so:
                # Epoch 0: 1139 / 10000
                # Epoch 1: 1136 / 10000
                # Epoch 2: 1135 / 10000
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # create a zero matrix determined by b's shape. (y by 1)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # create a zero matrix determined by w's shape. (y by x)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # x = training input
        # y = desired output
        # iterate through one batch
        for x, y in mini_batch:
            # perform backprop on b and w, i.e.,
            # compute partial derivatives delta(C_x)/delta(b^(l)_j) and delta (C_x)/delta(w^(l)_jk)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # nabla is a symbol indicating that it's a gradient vector
            # copy delta_nabla_b to nabla_b
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # copy delta_nabla_w to nabla_w
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update weights and biases based on the eta, the learning rate (between 0 and 100) (Formula (20) and (21))
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # create a zero matrix determined by b's shape.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # create a zero matrix determined by w's shape.
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        # x = the current training input, or the activation
        activation = x
        # create a list. first index = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # compute the dot product of matrices w and activation. lastly, add b
            # Formula (22): a' = delta(wa + b)
            z = np.dot(w, activation)+b
            # add it to list
            zs.append(z)
            # get the sigmoid of z
            activation = sigmoid(z)
            # add that to list
            activations.append(activation)
        # backward pass

        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # used for backpropagation. Pass in the last layer (output_activations), as well as
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    # Formula (3)
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

from scratch_handwritten_classifier import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])
net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
