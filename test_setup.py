"""Quick test to verify the Snake RL system is set up correctly"""

import sys
import os

# Add snake_rl to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from snake_rl.env.snake_env import SnakeEnv
        print("[OK] Snake environment")
        
        from snake_rl.agents.dqn.dqn_agent import DQNAgent
        from snake_rl.agents.dqn.q_network import QNetwork
        from snake_rl.agents.dqn.replay_buffer import ReplayBuffer
        print("[OK] DQN components")
        
        from snake_rl.agents.ppo.ppo_agent import PPOAgent
        from snake_rl.agents.ppo.actor_critic import ActorCritic
        print("[OK] PPO components")
        
        from snake_rl.utils.config import get_config
        from snake_rl.utils.logger import MetricsLogger
        from snake_rl.utils.scheduler import EpsilonScheduler, ClipScheduler
        print("[OK] Utilities")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_environment():
    """Test basic environment functionality"""
    print("\nTesting environment...")
    
    try:
        from snake_rl.env.snake_env import SnakeEnv
        
        env = SnakeEnv(grid_size=10, max_steps=200)
        state = env.reset()
        
        assert state.shape == (9,), f"Expected state shape (9,), got {state.shape}"
        print(f"[OK] Environment reset - state shape: {state.shape}")
        
        action = 1  # Go straight
        next_state, reward, done, info = env.step(action)
        
        assert next_state.shape == (9,), f"Expected next_state shape (9,), got {next_state.shape}"
        assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
        assert isinstance(done, bool), f"Expected bool done, got {type(done)}"
        print(f"[OK] Environment step - reward: {reward:.2f}, done: {done}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Environment test failed: {e}")
        return False


def test_dqn_agent():
    """Test DQN agent initialization and basic functionality"""
    print("\nTesting DQN agent...")
    
    try:
        from snake_rl.agents.dqn.dqn_agent import DQNAgent
        from snake_rl.utils.config import get_dqn_config
        import numpy as np
        
        config = get_dqn_config()
        agent = DQNAgent(state_size=9, action_size=3, config=config)
        
        print(f"[OK] DQN agent initialized on device: {agent.device}")
        
        # Test action selection
        state = np.random.randn(9).astype(np.float32)
        action = agent.select_action(state, epsilon=0.5)
        
        assert 0 <= action < 3, f"Invalid action: {action}"
        print(f"[OK] DQN action selection - action: {action}")
        
        return True
    except Exception as e:
        print(f"[FAIL] DQN test failed: {e}")
        return False


def test_ppo_agent():
    """Test PPO agent initialization and basic functionality"""
    print("\nTesting PPO agent...")
    
    try:
        from snake_rl.agents.ppo.ppo_agent import PPOAgent
        from snake_rl.utils.config import get_ppo_config
        import numpy as np
        
        config = get_ppo_config()
        agent = PPOAgent(state_size=9, action_size=3, config=config)
        
        print(f"[OK] PPO agent initialized on device: {agent.device}")
        
        # Test action selection
        state = np.random.randn(9).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        
        assert 0 <= action < 3, f"Invalid action: {action}"
        print(f"[OK] PPO action selection - action: {action}, value: {value:.2f}")
        
        return True
    except Exception as e:
        print(f"[FAIL] PPO test failed: {e}")
        return False


def test_innovations():
    """Test that innovations are implemented"""
    print("\nTesting innovations...")
    
    try:
        from snake_rl.env.snake_env import SnakeEnv
        
        env = SnakeEnv(grid_size=10, max_steps=200)
        state = env.reset()
        
        # Test Innovation 1: Distance shaping
        initial_distance = env.prev_distance
        print(f"[OK] Innovation 1 (Distance shaping) - Initial distance tracked: {initial_distance:.2f}")
        
        # Test Innovation 2: Loop detection
        assert hasattr(env, 'position_history'), "position_history not found"
        assert hasattr(env, 'steps_without_food'), "steps_without_food not found"
        print(f"[OK] Innovation 2 (Loop penalty) - Position history size: {len(env.position_history)}")
        
        # Test Innovation 3: Adaptive clipping
        from snake_rl.utils.scheduler import ClipScheduler
        scheduler = ClipScheduler(start=0.3, end=0.1, decay_rate=0.0001)
        clip_0 = scheduler.get_clip_epsilon(0)
        clip_1000 = scheduler.get_clip_epsilon(1000)
        assert clip_0 > clip_1000, "Clip should decay"
        print(f"[OK] Innovation 3 (Adaptive clipping) - Clip at 0: {clip_0:.3f}, at 1000: {clip_1000:.3f}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Innovations test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Snake RL System Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("DQN Agent", test_dqn_agent()))
    results.append(("PPO Agent", test_ppo_agent()))
    results.append(("Innovations", test_innovations()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("SUCCESS! All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Train DQN: python snake_rl/training/train_dqn.py")
        print("  2. Train PPO: python snake_rl/training/train_ppo.py")
        print("  3. Evaluate: python snake_rl/evaluation/evaluate_agent.py --agent ppo")
    else:
        print("WARNING: Some tests failed. Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

