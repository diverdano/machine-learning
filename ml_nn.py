#!usr/bin/python

# === load libraries ===
# key libraries
import numpy as np

# data prep

# data sets


# models

# metrics

# plot

# === objects ===

class Perceptron:
    """This class models an artificial neuron with step activation function."""
    def __init__(self, weights = np.array([1]), threshold = 0):
        """Initialize weights and threshold based on input arguments. Note that no type-checking is being performed here for simplicity."""
        self.weights = weights.astype(float)
        self.threshold = threshold
    def __repr__(self):
        return str({'weights':self.weights, 'threshold':self.threshold})
    def activate(self,inputs):
        """Takes in @param inputs, a list of numbers equal to length of weights. @return the output of a threshold perceptron with given inputs based on perceptron weights and threshold.""" 
        strength = np.dot(inputs,self.weights)
        return int(strength > self.threshold)
    def update(self, values, train, eta=.1):
        """Takes in a 2D array @param values consisting of a LIST of inputs and a 1D array @param train, consisting of a corresponding list of expected outputs.
        Updates internal weights according to the perceptron training rule using these values and an optional learning rate, @param eta."""
        iter_count = 0  ### ENUMVERATE!!!
        weights     = range(0, len(self.weights) - 1) # arrays start with zero
        # TODO: for each data point...
        for iteration in train:   # remember inputs is a list, output is scalar
            # TODO: obtain the neuron's prediction for that point
            prediction = self.activate(values[iter_count])        # obtain prediction
            error = train[iter_count] - prediction
            input_count = 0  ### ENUMVERATE!!!
            # TODO: update self.weights based on prediction accuracy, learning rate and input value
            for weight in self.weights:
                print(weight)
                self.weights[input_count] += eta * error * values[iter_count][input_count]
                input_count += 1
            iter_count += 1

# === data ===

Network = [[ Perceptron(np.array([1,1]),0), Perceptron(np.array([1,1]),1) ], [ Perceptron(np.array([1,-2]),0) ]] # [input perceptrons OR & AND],[output perceptron OR -2xAND]

# Part 2: Define a procedure to compute the output of the network, given inputs
def EvalNetwork(inputValues, Network):
    """Takes in @param inputValues, a list of input values, and @param Network that specifies a perceptron network. @return the output of the Network for the given set of inputs."""
    inputValues = np.array(list(inputValues))
    for layer in Network:
        results = [node.activate(inputValues) for node in layer]
        inputValues = results
    return results[0]

# === test functions ===

def test_eval():
    """A few tests to make sure that the perceptron class performs as expected."""
    print("0 XOR 0 = 0?:", EvalNetwork(np.array([0,0]), Network))
    print("0 XOR 1 = 1?:", EvalNetwork(np.array([0,1]), Network))
    print("1 XOR 0 = 1?:", EvalNetwork(np.array([1,0]), Network))
    print("1 XOR 1 = 0?:", EvalNetwork(np.array([1,1]), Network))

def test_update():
    """A few tests to make sure that the perceptron class performs as expected. Nothing should show up in the output if all the assertions pass."""
    def sum_almost_equal(array1, array2, tol = 1e-6):
        return sum(abs(array1 - array2)) < tol

    p1 = Perceptron(np.array([1,1,1]),0)
    p1.update(np.array([[2,0,-3]]), np.array([1]))
    print(p1.weights)
    assert sum_almost_equal(p1.weights, np.array([1.2, 1, 0.7]))

    p2 = Perceptron(np.array([1,2,3]),0)
    p2.update(np.array([[3,2,1],[4,0,-1]]),np.array([0,0]))
    assert sum_almost_equal(p2.weights, np.array([0.7, 1.8, 2.9]))

    p3 = Perceptron(np.array([3,0,2]),0)
    p3.update(np.array([[2,-2,4],[-1,-3,2],[0,2,1]]),np.array([0,1,0]))
    assert sum_almost_equal(p3.weights, np.array([2.7, -0.3, 1.7]))

def test_activation():
    """A few tests to make sure that the perceptron class performs as expected. Nothing should show up in the output if all the assertions pass."""
    p1 = Perceptron(np.array([1, 2]), 0.)
    assert p1.activate(np.array([ 1,-1])) == 0 # < threshold --> 0
    assert p1.activate(np.array([-1, 1])) == 1 # > threshold --> 1
    assert p1.activate(np.array([ 2,-1])) == 0 # on threshold --> 0

# === other functions ===

def activate(strength):
    # Try out different functions here. Input strength will be a number, with
    # another number as output.
    return np.power(strength,2)

def activation_derivative(activate, strength):
    #numerically approximate
    return (activate(strength+1e-5)-activate(strength-1e-5))/(2e-5)
