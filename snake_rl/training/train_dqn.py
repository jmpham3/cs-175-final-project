"""Training script for DQN agent"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.agents.dqn.dqn_agent import DQNAgent
from snake_rl.utils.config import get_config
from snake_rl.utils.logger import MetricsLogger
from snake_rl.utils.scheduler import EpsilonScheduler


def train_dqn(num_episodes: int = 2000):
    """
    Train DQN agent on Snake environment
    
    Args:
        num_episodes: Number of episodes to train
    """
    # Get configuration
    config = get_config()
    env_config = config['env']
    dqn_config = config['dqn']
    training_config = config['training']
    
    # Initialize environment
    env = SnakeEnv(
        grid_size=env_config['grid_size'],
        max_steps=env_config['max_steps']
    )
    
    # Initialize agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=dqn_config
    )
    
    # Initialize epsilon scheduler
    epsilon_scheduler = EpsilonScheduler(
        start=dqn_config['epsilon_start'],
        end=dqn_config['epsilon_end'],
        decay_steps=dqn_config['epsilon_decay_steps']
    )
    
    # Initialize logger
    logger = MetricsLogger(log_frequency=training_config['log_frequency'])
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join('snake_rl', 'models'), exist_ok=True)
    
    print("=" * 60)
    print("Training DQN Agent on Snake Environment")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Grid Size: {env_config['grid_size']}x{env_config['grid_size']}")
    print(f"State Size: {env.state_size}")
    print(f"Action Size: {env.action_size}")
    print(f"Device: {agent.device}")
    print("=" * 60)
    
    # Training loop
    total_steps = 0
    best_avg_reward = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_steps = 0
        
        done = False
        while not done:
            # Select action
            epsilon = epsilon_scheduler.get_epsilon(total_steps)
            action = agent.select_action(state, epsilon)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            episode_loss += loss
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
        
        # Log metrics
        avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
        logger.log(episode, {
            'reward': episode_reward,
            'score': env.score,
            'steps': episode_steps,
            'epsilon': epsilon,
            'loss': avg_loss,
            'avg_reward_100': logger.get_moving_average('reward', window=100)
        })
        
        # Save checkpoint periodically
        if episode % training_config['save_frequency'] == 0:
            model_path = os.path.join('snake_rl', 'models', f'dqn_episode_{episode}.pt')
            agent.save(model_path)
            
            # Save best model
            avg_reward = logger.get_moving_average('reward', window=100)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_model_path = os.path.join('snake_rl', 'models', 'saved_dqn.pt')
                agent.save(best_model_path)
    
    # Save final model
    final_model_path = os.path.join('snake_rl', 'models', 'saved_dqn.pt')
    agent.save(final_model_path)
    
    # Save training log
    log_path = os.path.join('snake_rl', 'models', 'dqn_training_log.csv')
    logger.save_to_csv(log_path)
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Final Average Reward (100 episodes): {logger.get_moving_average('reward', window=100):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent on Snake')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of episodes to train (default: 2000)')
    
    args = parser.parse_args()
    
    train_dqn(num_episodes=args.episodes)

