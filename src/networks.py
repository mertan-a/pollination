from torch import nn
from torch.nn.utils import parameters_to_vector
from torch import sigmoid

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return sigmoid(logits) - 0.4

class NeuralNetworkBig(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetworkBig, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return sigmoid(logits) - 0.4

# test code
if __name__ == '__main__':
    mlp = NeuralNetwork(input_size=25*9, output_size=25)
    print(f'num params: {parameters_to_vector(mlp.parameters()).shape}')
    mlp = NeuralNetworkBig(input_size=25*9, output_size=25)
    print(f'num params: {parameters_to_vector(mlp.parameters()).shape}')





