"""Configuration system for Snake RL"""

def get_config():
    """Returns the complete configuration dictionary for the Snake RL system"""
    
    config = {
        # Environment configuration
        'env': {
            'grid_size': 10,
            'max_steps': 200,
            'render_fps': 10,
        },
        
        # DQN configuration
        'dqn': {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay_steps': 10000,
            'replay_buffer_size': 10000,
            'batch_size': 64,
            'target_update_frequency': 100,
            'hidden_dim': 128,
        },
        
        # PPO configuration
        'ppo': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon_start': 0.3,
            'clip_epsilon_end': 0.1,
            'clip_decay_rate': 0.0001,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'n_epochs': 10,
            'batch_size': 64,
            'rollout_steps': 2048,
            'hidden_dim': 128,
        },
        
        # Training configuration
        'training': {
            'default_episodes': 2000,
            'save_frequency': 500,
            'log_frequency': 100,
        }
    }
    
    return config


def get_env_config():
    """Get environment configuration"""
    return get_config()['env']


def get_dqn_config():
    """Get DQN configuration"""
    return get_config()['dqn']


def get_ppo_config():
    """Get PPO configuration"""
    return get_config()['ppo']


def get_training_config():
    """Get training configuration"""
    return get_config()['training']


