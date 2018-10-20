import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Neural network that takes environment states as inputs, and outputs utility (or values) for every possible action
    to perform in the next step. Code adapted from Udacity Deep Reinforcement Learning Nanodegree course (http://www.udacity.com)"""

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128):
        """
        Parameters
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first (fully connected) hidden layer
            fc2_units (int): Number of nodes in second (fully connected) hidden layer
        """
        super(QNetwork, self).__init__()

        # Build network components
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Computes forward pass of network, outputing action values"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self):
        """Save model parameters to file 'checkpoint.pth'"""
        torch.save(self.state_dict(), 'checkpoint.pth')

    def load(self, file = 'checkpoint.pth'):
        """If possible, load model parameters from file (extension .pth)"""
        try:
            state_dict = torch.load(file)
        except FileNotFoundError as err:
            raise Exception('No file named "checkpoint.pth" was found')
        else:
            self.load_state_dict(state_dict)
