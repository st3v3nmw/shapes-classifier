# Mwangi Stephen, Mutuku Moses, Mokaya Sharon
import codecs
import json
import numpy as np


class NeuralNetwork:
    """
        Neural Network Class
        A single layer network (one hidden layer)
        Takes no. of input nodes, number of hidden nodes, number of output nodes and the learning rate during initialization
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        # initialize weights using random normally distributed values
        self.weights_ih = np.random.normal(0.0, pow(self.n_hidden, -0.5), (self.n_hidden, self.n_inputs))
        self.weights_ho = np.random.normal(0.0, pow(self.n_outputs, -0.5), (self.n_outputs, self.n_hidden))

    @staticmethod
    def sigmoid(X):
        # sigmoid activation function
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def load_from_file(filepath):
        # static method to load a saved neural network model from a file
        model = json.loads(codecs.open(filepath).read())
        n_inputs, n_hidden, n_outputs, learning_rate = model[0]
        nn = NeuralNetwork(n_inputs, n_hidden, n_outputs, learning_rate)
        nn.weights_ih = np.array(model[1])
        nn.weights_ho = np.array(model[2])
        return nn

    def accuracy(self, data_, labels_):
        # function to calculate the accuracy of the neural network given certain inputs and labels
        err = 0
        total = len(data_)
        for i_ in range(total):
            item_ = data_[i_]
            pred = np.argmax(self.predict(item_))
            if pred != np.argmax(labels_[i_]):
                err += 1
        return 1 - err / total

    def write_to_file(self, filepath):
        # function to save the neural network model to a file in json format
        config = [self.n_inputs, self.n_hidden, self.n_outputs, self.learning_rate]
        model = [config, self.weights_ih.tolist(), self.weights_ho.tolist()]
        json.dump(model, codecs.open(filepath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=1)

    def predict(self, input_list, returnLayerOutputs=False):
        # function to feed forward the inputs and return a classification at the end
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        if returnLayerOutputs:
            return [final_outputs, hidden_outputs, inputs]
        else:
            return final_outputs.ravel()

    def train(self, input_list, target_list):
        # function to train the network by adjusting the connection weights using gradient descent (backpropagation algorithm)
        targets = np.array(target_list, ndmin=2).T
        final_outputs, hidden_outputs, inputs = self.predict(input_list, True)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)
        self.weights_ho += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.weights_ih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
