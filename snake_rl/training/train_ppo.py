"""Training script for PPO agent"""

import sys
import os
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.agents.ppo.ppo_agent import PPOAgent
from snake_rl.utils.config import get_config
from snake_rl.utils.logger import MetricsLogger
from snake_rl.utils.scheduler import ClipScheduler


def collect_rollouts(env, agent, num_steps):
    """
    Collect rollout data from environment
    
    Args:
        env: Snake environment
        agent: PPO agent
        num_steps: Number of steps to collect
        
    Returns:
        Dictionary of rollout data
    """
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    
    state = env.reset()
    episode_reward = 0
    episode_count = 0
    
    # Track all completed episode rewards and scores
    episode_rewards = []
    episode_scores = []
    
    for _ in range(num_steps):
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store data
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        
        episode_reward += reward
        
        state = next_state
        
        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)  # Save this episode's reward
            episode_scores.append(env.score)        # Save this episode's score
            episode_reward = 0                      # Reset for next episode
            state = env.reset()
    
    # Calculate average from completed episodes
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    avg_score = np.mean(episode_scores) if episode_scores else 0.0
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'log_probs': np.array(log_probs),
        'values': values,
        'rewards': rewards,
        'dones': dones,
        'episode_reward': avg_reward,
        'episode_score': avg_score,
        'episode_count': episode_count
    }


def train_ppo(num_episodes: int = 2000):
    """
    Train PPO agent on Snake environment
    
    Args:
        num_episodes: Number of episodes to train
    """
    # Get configuration
    config = get_config()
    env_config = config['env']
    ppo_config = config['ppo']
    training_config = config['training']
    
    # Initialize environment
    env = SnakeEnv(
        grid_size=env_config['grid_size'],
        max_steps=env_config['max_steps']
    )
    
    # Initialize agent
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=ppo_config
    )
    
    # Initialize clip scheduler
    clip_scheduler = ClipScheduler(
        start=ppo_config['clip_epsilon_start'],
        end=ppo_config['clip_epsilon_end'],
        decay_rate=ppo_config['clip_decay_rate']
    )
    
    # Initialize logger
    logger = MetricsLogger(log_frequency=training_config['log_frequency'])
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join('snake_rl', 'models'), exist_ok=True)
    
    print("=" * 60)
    print("Training PPO Agent on Snake Environment")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Grid Size: {env_config['grid_size']}x{env_config['grid_size']}")
    print(f"State Size: {env.state_size}")
    print(f"Action Size: {env.action_size}")
    print(f"Rollout Steps: {ppo_config['rollout_steps']}")
    print(f"Device: {agent.device}")
    print("=" * 60)
    
    # Training loop
    episode_count = 0
    best_avg_reward = -float('inf')
    
    while episode_count < num_episodes:
        # Collect rollouts
        rollout = collect_rollouts(env, agent, ppo_config['rollout_steps'])
        
        # Get current clip epsilon (Innovation 3: Adaptive clipping)
        clip_epsilon = clip_scheduler.get_clip_epsilon(agent.update_count)
        
        # Compute advantages and returns using GAE
        advantages, returns = agent.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones']
        )
        
        # Update policy
        loss_info = agent.update(
            rollout['states'],
            rollout['actions'],
            rollout['log_probs'],
            np.array(advantages),
            np.array(returns),
            clip_epsilon
        )
        
        # Update episode count
        episode_count += rollout['episode_count']
        
        # Log metrics
        if episode_count > 0:
            logger.log(episode_count, {
                'reward': rollout['episode_reward'],
                'score': rollout['episode_score'],
                'policy_loss': loss_info['policy_loss'],
                'value_loss': loss_info['value_loss'],
                'entropy': loss_info['entropy'],
                'clip_epsilon': clip_epsilon,
                'avg_reward_100': logger.get_moving_average('reward', window=100)
            })
        
        # Save checkpoint periodically
        if episode_count % training_config['save_frequency'] == 0 and episode_count > 0:
            model_path = os.path.join('snake_rl', 'models', f'ppo_episode_{episode_count}.pt')
            agent.save(model_path)
            
            # Save best model
            avg_reward = logger.get_moving_average('reward', window=100)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_model_path = os.path.join('snake_rl', 'models', 'saved_ppo.pt')
                agent.save(best_model_path)
    
    # Save final model
    final_model_path = os.path.join('snake_rl', 'models', 'saved_ppo.pt')
    agent.save(final_model_path)
    
    # Save training log
    log_path = os.path.join('snake_rl', 'models', 'ppo_training_log.csv')
    logger.save_to_csv(log_path)
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Final Average Reward (100 episodes): {logger.get_moving_average('reward', window=100):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent on Snake')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of episodes to train (default: 2000)')
    
    args = parser.parse_args()
    
    train_ppo(num_episodes=args.episodes)

