"""PPO Agent implementation with adaptive clipping"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

from .actor_critic import ActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization agent with adaptive clipping schedule
    """
    
    def __init__(self, state_size: int, action_size: int, config: Dict):
        """
        Initialize PPO agent
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            config: Configuration dictionary
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        hidden_dim = config.get('hidden_dim', 128)
        
        # Network
        self.network = ActorCritic(state_size, action_size, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Update counter for adaptive clipping
        self.update_count = 0
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action from policy
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
            return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            advantages: List of advantage estimates
            returns: List of return estimates
        """
        advantages = []
        returns = []
        gae = 0
        
        # Work backwards from the end
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
            # Return (for value function target)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray,
               advantages: np.ndarray, returns: np.ndarray, clip_epsilon: float) -> Dict[str, float]:
        """
        Update policy using PPO with adaptive clipping
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old log probabilities
            advantages: Batch of advantages
            returns: Batch of returns
            clip_epsilon: Current clipping epsilon value
            
        Returns:
            Dictionary of loss metrics
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # Multiple epochs of updates
        for _ in range(self.n_epochs):
            # Evaluate current policy
            log_probs, values, entropy = self.network.evaluate_actions(states, actions)
            
            # Policy loss with clipping
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        self.update_count += 1
        
        return {
            'policy_loss': total_policy_loss / self.n_epochs,
            'value_loss': total_value_loss / self.n_epochs,
            'entropy': total_entropy / self.n_epochs,
        }
    
    def save(self, filepath: str):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        print(f"Model loaded from {filepath}")


