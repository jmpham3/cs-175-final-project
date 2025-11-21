"""Learning rate and parameter schedulers"""

import math


class EpsilonScheduler:
    """Linear epsilon decay scheduler for DQN exploration"""
    
    def __init__(self, start: float, end: float, decay_steps: int):
        """
        Initialize epsilon scheduler
        
        Args:
            start: Initial epsilon value
            end: Final epsilon value
            decay_steps: Number of steps to decay from start to end
        """
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        
    def get_epsilon(self, step: int) -> float:
        """
        Get epsilon value for current step
        
        Args:
            step: Current training step
            
        Returns:
            Current epsilon value
        """
        if step >= self.decay_steps:
            return self.end
        
        # Linear interpolation
        progress = step / self.decay_steps
        return self.start + (self.end - self.start) * progress


class ClipScheduler:
    """Exponential decay scheduler for PPO clipping epsilon"""
    
    def __init__(self, start: float, end: float, decay_rate: float):
        """
        Initialize clip scheduler
        
        Args:
            start: Initial clip epsilon value
            end: Minimum clip epsilon value
            decay_rate: Exponential decay rate
        """
        self.start = start
        self.end = end
        self.decay_rate = decay_rate
        
    def get_clip_epsilon(self, update: int) -> float:
        """
        Get clip epsilon value for current update
        
        Args:
            update: Current update iteration
            
        Returns:
            Current clip epsilon value
        """
        # Exponential decay: clip = max(end, start * exp(-decay_rate * update))
        decayed_value = self.start * math.exp(-self.decay_rate * update)
        return max(self.end, decayed_value)


