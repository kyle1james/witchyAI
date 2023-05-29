import numpy as np
import matplotlib.pyplot as plt
from neuralNet import BasicNeuralNet

class GraphingNeuralNet(BasicNeuralNet):
    '''
    🏮: wrapper for visuals of a neural net
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.training_errors = []

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
                super().train_one_example(input_vector, target_vector, learning_rate)
                output = self.feedforward(input_vector)
                output_errors = self._preprocess_input_data(target_vector) - output
                self.training_errors.append(np.mean((output_errors ** 2)))

    def plot_training_error(self):
        plt.plot(self.training_errors)
        plt.xlabel('Training iterations')
        plt.ylabel('Mean squared error')
        plt.title('Training error')
        plt.show()

    def plot_output(self, input_data):
        outputs = [self.feedforward(input_vector).flatten() for input_vector in input_data]
        x = range(len(input_data))
        plt.plot(x, outputs, 'o-')
        plt.xlabel('Input data index')
        plt.ylabel('Output')
        plt.title('Neural network output')
        plt.show()

    def testNet(self,input_data,target_data):
        self.train_epochs(input_data, target_data, epochs=5000)
        self.plot_training_error()
        self.plot_output(input_data)

if __name__ == "__main__":
    # Example usage of GraphingNeuralNet
    nn = GraphingNeuralNet(3, 4, 2)
    input_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    target_data = [[0.4, 0.8], [0.6, 0.4], [0.8, 0.2]]
    nn.testNet(input_data, target_data)
