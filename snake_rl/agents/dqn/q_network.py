"""Q-Network for DQN agent"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Fully connected Q-network for DQN
    Maps state to Q-values for each action
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_dim: int = 128):
        """
        Initialize Q-network
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            state: State tensor [batch_size, state_size]
            
        Returns:
            Q-values for each action [batch_size, action_size]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


