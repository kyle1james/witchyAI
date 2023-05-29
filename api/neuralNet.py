import numpy as np

class BasicNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with given layer sizes and random weights.
        
        :param input_size: int, size of the input layer
        :param hidden_size: int, size of the hidden layer
        :param output_size: int, size of the output layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with random values
        self.weights_ih = np.random.randn(hidden_size, input_size)
        self.weights_ho = np.random.randn(output_size, hidden_size)
        self.bias_h = np.random.randn(hidden_size, 1)
        self.bias_o = np.random.randn(output_size, 1)

    '''
    makes the method independent of any instance or class state
    more better for computer bip-boops
    '''
    @staticmethod
    def sigmoid(x):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    

    @staticmethod
    def sigmoid_derivative(x):
        """Compute the derivative of the sigmoid activation function."""
        return x * (1 - x)

    def _preprocess_input_data(self, input_data):
        """
        Ensure input data is in the correct shape (input_size, 1).
        
        :param input_data: list or np.array, input data (1D or matrix)
        :return: np.array, input data in the correct shape (input_size, 1)
        """
        return np.array(input_data, ndmin=2).T

    def feedforward(self, input_data):
        """
        Perform the feedforward process on the input data.
        
        :param input_data: list or np.array, input data (1D or matrix)
        :return: np.array, output of the neural network
        """
        
        input_data = self._preprocess_input_data(input_data)

        # Calculate hidden layer output
        hidden_output = self.sigmoid(np.dot(self.weights_ih, input_data) + self.bias_h)

        # Calculate output layer output
        output = self.sigmoid(np.dot(self.weights_ho, hidden_output) + self.bias_o)

        return output

    def train_one_example(self, input_data, target_data, learning_rate=0.1):
        """
        Train the neural network using one example of input and target data.

        :param input_data: list or np.array, input data (1D or matrix)
        :param target_data: list or np.array, target data (1D or matrix)
        :param learning_rate: float, learning rate for the neural network
        """
        input_data = self._preprocess_input_data(input_data)
        target_data = self._preprocess_input_data(target_data)

        # Perform feedforward to get hidden and output layer outputs
        hidden_output = self.sigmoid(np.dot(self.weights_ih, input_data) + self.bias_h)
        output = self.sigmoid(np.dot(self.weights_ho, hidden_output) + self.bias_o)

        # Calculate output layer errors
        output_errors = target_data - output

        # Calculate hidden layer errors
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        # Update output layer weights and biases
        self.weights_ho += learning_rate * np.dot(output_errors * self.sigmoid_derivative(output), hidden_output.T)
        self.bias_o += learning_rate * output_errors * self.sigmoid_derivative(output)

        # Update hidden layer weights and biases
        self.weights_ih += learning_rate * np.dot(hidden_errors * self.sigmoid_derivative(hidden_output), input_data.T)
        self.bias_h += learning_rate * hidden_errors * self.sigmoid_derivative(hidden_output)

    def train_epochs(self, input_data, target_data, epochs, learning_rate=0.1):
        """
        Train the neural network for a specified number of epochs.
        
        :param input_data: list of list or np.array, input data (1D or matrix)
        :param target_data: list of list or np.array, target data (1D or matrix)
        :param epochs: int, number of epochs to train the neural network
        :param learning_rate: float, learning rate for the neural network
        """
        for epoch in range(epochs):
            for input_vector, target_vector in zip(input_data, target_data):
                self.train_one_example(input_vector, target_vector, learning_rate)

    def predict(self, input_data):
        """
        Predict the output for the given input data using the neural network.

        :param input_data: list of list or np.array, input data (1D or matrix)
        :return: list of list, predicted output for each input data
        """
        input_data = np.array(input_data).T
        predictions = [self.feedforward(input_data[:, i]).flatten().tolist() for i in range(input_data.shape[1])]
        return predictions

if __name__ == "__main__":
    # Example usage of BasicNeuralNet
    # Create a neural network with 3 input nodes, 4 hidden nodes, and 2 output nodes
    nn = BasicNeuralNet(3, 4, 2)

    # Train the neural network with input data and target data
    input_data = [[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6],
                  [0.7, 0.8, 0.9]]

    target_data = [[0.4, 0.8],
                   [0.6, 0.4],
                   [0.8, 0.2]]
    
    nn.train_epochs(input_data, target_data, epochs=5000)

    # Example usage of the predict method
    predictions = nn.predict(input_data)
    print(predictions)