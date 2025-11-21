"""Plot learning curves from training logs"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def moving_average(data, window=100):
    """Calculate moving average"""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_dqn_curves(log_path: str, output_dir: str):
    """
    Plot DQN training curves
    
    Args:
        log_path: Path to DQN training log CSV
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(log_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Curves', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward')
    if len(df) > 100:
        ma = moving_average(df['reward'].values, window=100)
        axes[0, 0].plot(df['episode'].values[99:], ma, label='100-Episode Moving Average', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode scores
    axes[0, 1].plot(df['episode'], df['score'], alpha=0.3, label='Episode Score')
    if len(df) > 100:
        ma = moving_average(df['score'].values, window=100)
        axes[0, 1].plot(df['episode'].values[99:], ma, label='100-Episode Moving Average', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Episode Scores (Food Eaten)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(df['episode'], df['steps'], alpha=0.3, label='Episode Length')
    if len(df) > 100:
        ma = moving_average(df['steps'].values, window=100)
        axes[1, 0].plot(df['episode'].values[99:], ma, label='100-Episode Moving Average', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode Lengths (Survival Time)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epsilon decay
    axes[1, 1].plot(df['episode'], df['epsilon'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Epsilon Decay Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'dqn_learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"DQN learning curves saved to {save_path}")
    
    plt.close()


def plot_ppo_curves(log_path: str, output_dir: str):
    """
    Plot PPO training curves
    
    Args:
        log_path: Path to PPO training log CSV
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(log_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PPO Training Curves', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward')
    if len(df) > 100:
        ma = moving_average(df['reward'].values, window=100)
        axes[0, 0].plot(df['episode'].values[99:], ma, label='100-Episode Moving Average', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode scores
    axes[0, 1].plot(df['episode'], df['score'], alpha=0.3, label='Episode Score')
    if len(df) > 100:
        ma = moving_average(df['score'].values, window=100)
        axes[0, 1].plot(df['episode'].values[99:], ma, label='100-Episode Moving Average', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Episode Scores (Food Eaten)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Clip epsilon (Innovation 3)
    axes[0, 2].plot(df['episode'], df['clip_epsilon'])
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Clip Epsilon')
    axes[0, 2].set_title('Adaptive PPO Clipping Schedule')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Policy loss
    axes[1, 0].plot(df['episode'], df['policy_loss'], alpha=0.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Policy Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value loss
    axes[1, 1].plot(df['episode'], df['value_loss'], alpha=0.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Value Loss')
    axes[1, 1].set_title('Value Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 2].plot(df['episode'], df['entropy'], alpha=0.5)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Entropy')
    axes[1, 2].set_title('Policy Entropy')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'ppo_learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PPO learning curves saved to {save_path}")
    
    plt.close()


def plot_comparison(dqn_log_path: str, ppo_log_path: str, output_dir: str):
    """
    Plot comparison between DQN and PPO
    
    Args:
        dqn_log_path: Path to DQN training log CSV
        ppo_log_path: Path to PPO training log CSV
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    dqn_df = pd.read_csv(dqn_log_path)
    ppo_df = pd.read_csv(ppo_log_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('DQN vs PPO Comparison', fontsize=16)
    
    # Episode rewards
    if len(dqn_df) > 100:
        dqn_ma = moving_average(dqn_df['reward'].values, window=100)
        axes[0].plot(dqn_df['episode'].values[99:], dqn_ma, label='DQN', linewidth=2)
    
    if len(ppo_df) > 100:
        ppo_ma = moving_average(ppo_df['reward'].values, window=100)
        axes[0].plot(ppo_df['episode'].values[99:], ppo_ma, label='PPO', linewidth=2)
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (100-Episode MA)')
    axes[0].set_title('Reward Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Episode scores
    if len(dqn_df) > 100:
        dqn_ma = moving_average(dqn_df['score'].values, window=100)
        axes[1].plot(dqn_df['episode'].values[99:], dqn_ma, label='DQN', linewidth=2)
    
    if len(ppo_df) > 100:
        ppo_ma = moving_average(ppo_df['score'].values, window=100)
        axes[1].plot(ppo_df['episode'].values[99:], ppo_ma, label='PPO', linewidth=2)
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score (100-Episode MA)')
    axes[1].set_title('Score Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'dqn_vs_ppo_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot learning curves from training logs')
    parser.add_argument('--agent', type=str, choices=['dqn', 'ppo', 'both'], default='both',
                       help='Which agent logs to plot (default: both)')
    parser.add_argument('--output-dir', type=str, default='snake_rl/plots',
                       help='Directory to save plots (default: snake_rl/plots)')
    parser.add_argument('--dqn-log', type=str, default='snake_rl/models/dqn_training_log.csv',
                       help='Path to DQN training log')
    parser.add_argument('--ppo-log', type=str, default='snake_rl/models/ppo_training_log.csv',
                       help='Path to PPO training log')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Plotting Learning Curves")
    print("=" * 60)
    
    # Plot DQN
    if args.agent in ['dqn', 'both']:
        if os.path.exists(args.dqn_log):
            plot_dqn_curves(args.dqn_log, args.output_dir)
        else:
            print(f"Warning: DQN log not found at {args.dqn_log}")
    
    # Plot PPO
    if args.agent in ['ppo', 'both']:
        if os.path.exists(args.ppo_log):
            plot_ppo_curves(args.ppo_log, args.output_dir)
        else:
            print(f"Warning: PPO log not found at {args.ppo_log}")
    
    # Plot comparison
    if args.agent == 'both':
        if os.path.exists(args.dqn_log) and os.path.exists(args.ppo_log):
            plot_comparison(args.dqn_log, args.ppo_log, args.output_dir)
    
    print("=" * 60)
    print("Plotting Complete!")
    print("=" * 60)

