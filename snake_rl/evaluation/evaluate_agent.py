"""Evaluation script for trained agents"""

import sys
import os
import argparse
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.agents.dqn.dqn_agent import DQNAgent
from snake_rl.agents.ppo.ppo_agent import PPOAgent
from snake_rl.utils.config import get_config

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Rendering will be disabled.")


def render_pygame(env, screen, cell_size):
    """Render the game using pygame"""
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    DARK_GREEN = (0, 150, 0)
    
    screen.fill(BLACK)
    
    # Draw grid
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, WHITE, rect, 1)
    
    # Draw snake
    for i, (x, y) in enumerate(env.snake):
        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
        color = GREEN if i == 0 else DARK_GREEN
        pygame.draw.rect(screen, color, rect)
    
    # Draw food
    if env.food:
        food_x, food_y = env.food
        rect = pygame.Rect(food_x * cell_size, food_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, RED, rect)
    
    pygame.display.flip()


def evaluate_agent(agent_type: str, model_path: str, num_episodes: int = 100, render: bool = False):
    """
    Evaluate a trained agent
    
    Args:
        agent_type: Type of agent ('dqn' or 'ppo')
        model_path: Path to model checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game
    """
    # Get configuration
    config = get_config()
    env_config = config['env']
    
    # Initialize environment
    env = SnakeEnv(
        grid_size=env_config['grid_size'],
        max_steps=env_config['max_steps']
    )
    
    # Initialize agent
    if agent_type.lower() == 'dqn':
        agent_config = config['dqn']
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            config=agent_config
        )
    elif agent_type.lower() == 'ppo':
        agent_config = config['ppo']
        agent = PPOAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            config=agent_config
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    agent.load(model_path)
    
    # Initialize pygame if rendering
    screen = None
    clock = None
    if render and PYGAME_AVAILABLE:
        pygame.init()
        cell_size = 40
        screen_size = env.grid_size * cell_size
        screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption(f"Snake - {agent_type.upper()} Agent")
        clock = pygame.time.Clock()
    
    print("=" * 60)
    print(f"Evaluating {agent_type.upper()} Agent")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print("=" * 60)
    
    # Evaluation loop
    scores = []
    survival_times = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action (no exploration)
            if agent_type.lower() == 'dqn':
                action = agent.select_action(state, epsilon=0.0)
            else:  # PPO
                action, _, _ = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            # Render
            if render and screen is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        pygame.quit()
                        return
                
                render_pygame(env, screen, 40)
                clock.tick(env_config['render_fps'])
        
        scores.append(env.score)
        survival_times.append(env.steps)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - "
                  f"Score: {env.score} - "
                  f"Steps: {env.steps} - "
                  f"Avg Score: {np.mean(scores):.2f}")
    
    # Print statistics
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    print(f"Mean Survival Time: {np.mean(survival_times):.2f} ± {np.std(survival_times):.2f}")
    print("=" * 60)
    
    if render and PYGAME_AVAILABLE:
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained agent on Snake')
    parser.add_argument('--agent', type=str, required=True, choices=['dqn', 'ppo'],
                       help='Type of agent to evaluate')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (default: snake_rl/models/saved_{agent}.pt)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--render', action='store_true',
                       help='Render the game during evaluation')
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model is None:
        args.model = os.path.join('snake_rl', 'models', f'saved_{args.agent}.pt')
    
    evaluate_agent(args.agent, args.model, args.episodes, args.render)


