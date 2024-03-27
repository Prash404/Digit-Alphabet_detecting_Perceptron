import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))
    
    def train(self, X, y, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward propagation
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output_layer_output = sigmoid(output_layer_input)
            # Calculate loss (MSE)
            loss = np.mean((y - output_layer_output) ** 2)
            losses.append(loss)
            # Backpropagation
            output_error = y - output_layer_output
            output_delta = output_error * sigmoid_derivative(output_layer_output)

            hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta) * self.learning_rate
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
            self.weights_input_hidden += X.T.dot(hidden_layer_delta) * self.learning_rate
            self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate
        return losses