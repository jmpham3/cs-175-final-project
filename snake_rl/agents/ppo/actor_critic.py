"""Actor-Critic network for PPO"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared base layers
    Actor outputs action probabilities, Critic outputs value estimate
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_dim: int = 128):
        """
        Initialize Actor-Critic network
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor):
        """
        Forward pass through network
        
        Args:
            state: State tensor [batch_size, state_size]
            
        Returns:
            action_probs: Action probability distribution
            value: State value estimate
        """
        features = self.base(state)
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def get_action_and_value(self, state: torch.Tensor):
        """
        Get action, log probability, entropy, and value
        
        Args:
            state: State tensor
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value estimate
        """
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions for given states
        
        Args:
            states: State tensor [batch_size, state_size]
            actions: Action tensor [batch_size]
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Entropy of action distributions
        """
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy


