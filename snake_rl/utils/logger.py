"""Metrics logging utilities"""

import csv
import os
from typing import Dict, Any, List


class MetricsLogger:
    """Logger for tracking and saving training metrics"""
    
    def __init__(self, log_frequency: int = 100):
        """
        Initialize the metrics logger
        
        Args:
            log_frequency: How often to print logs to console
        """
        self.log_frequency = log_frequency
        self.metrics_history: List[Dict[str, Any]] = []
        self.episode_count = 0
        
    def log(self, episode: int, metrics: Dict[str, Any]):
        """
        Log metrics for a given episode
        
        Args:
            episode: Episode number
            metrics: Dictionary of metric names to values
        """
        self.episode_count = episode
        metrics_with_episode = {'episode': episode, **metrics}
        self.metrics_history.append(metrics_with_episode)
        
        # Print to console periodically
        if episode % self.log_frequency == 0:
            self._print_metrics(metrics_with_episode)
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Pretty print metrics to console"""
        metric_strs = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                      for k, v in metrics.items()]
        print(" | ".join(metric_strs))
    
    def save_to_csv(self, filepath: str):
        """
        Save all logged metrics to a CSV file
        
        Args:
            filepath: Path to save the CSV file
        """
        if not self.metrics_history:
            print(f"No metrics to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write to CSV
        fieldnames = list(self.metrics_history[0].keys())
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)
        
        print(f"Metrics saved to {filepath}")
    
    def get_latest_metric(self, metric_name: str) -> Any:
        """Get the latest value of a specific metric"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1].get(metric_name)
    
    def get_moving_average(self, metric_name: str, window: int = 100) -> float:
        """
        Calculate moving average of a metric
        
        Args:
            metric_name: Name of the metric
            window: Window size for moving average
            
        Returns:
            Moving average value
        """
        if not self.metrics_history:
            return 0.0
        
        recent_values = [m[metric_name] for m in self.metrics_history[-window:] 
                        if metric_name in m]
        
        if not recent_values:
            return 0.0
        
        return sum(recent_values) / len(recent_values)


