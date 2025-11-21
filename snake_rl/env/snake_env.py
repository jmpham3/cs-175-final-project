"""Snake environment with Gym-like API and custom innovations"""

import numpy as np
import random
from collections import deque
from typing import Tuple, Dict, Any


class SnakeEnv:
    """
    Snake game environment with reinforcement learning innovations:
    1. Distance-based shaping reward
    2. Anti-loop / no-progress penalty
    3. Gym-like API (reset, step)
    """
    
    # Actions: 0 = turn left, 1 = straight, 2 = turn right
    ACTION_LEFT = 0
    ACTION_STRAIGHT = 1
    ACTION_RIGHT = 2
    
    # Directions: 0 = up, 1 = right, 2 = down, 3 = left
    DIR_UP = 0
    DIR_RIGHT = 1
    DIR_DOWN = 2
    DIR_LEFT = 3
    
    def __init__(self, grid_size: int = 10, max_steps: int = 200):
        """
        Initialize Snake environment
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            max_steps: Maximum steps before episode terminates
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Game state
        self.snake = []  # List of (x, y) positions
        self.direction = self.DIR_RIGHT
        self.food = None
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        
        # Innovation: Track recent positions for loop detection
        self.position_history = deque(maxlen=8)
        
        # Previous distance to food (for distance shaping)
        self.prev_distance = 0
        
        # State space size
        self.state_size = 9  # 2 (food direction) + 3 (danger) + 4 (direction one-hot)
        self.action_size = 3
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial state vector
        """
        # Initialize snake at center, length 3
        center = self.grid_size // 2
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center)
        ]
        self.direction = self.DIR_RIGHT
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.position_history.clear()
        
        # Place food
        self._place_food()
        
        # Initialize distance
        self.prev_distance = self._get_distance_to_food()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0=left, 1=straight, 2=right)
            
        Returns:
            next_state: Next state vector
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.steps += 1
        self.steps_without_food += 1
        
        # Update direction based on action
        self._update_direction(action)
        
        # Move snake
        head_x, head_y = self.snake[0]
        if self.direction == self.DIR_UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == self.DIR_DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == self.DIR_LEFT:
            new_head = (head_x - 1, head_y)
        else:  # RIGHT
            new_head = (head_x + 1, head_y)
        
        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return self._get_state(), -10.0, True, {'reason': 'wall_collision'}
        
        # Check collision with self
        if new_head in self.snake:
            return self._get_state(), -10.0, True, {'reason': 'self_collision'}
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Calculate reward
        reward = 0.0
        done = False
        info = {}
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            reward += 10.0
            self._place_food()
            info['ate_food'] = True
        else:
            # Remove tail if not eaten
            self.snake.pop()
            info['ate_food'] = False
        
        # Innovation 1: Distance-based shaping reward
        current_distance = self._get_distance_to_food()
        if current_distance < self.prev_distance:
            reward += 0.1  # Getting closer
        elif current_distance > self.prev_distance:
            reward -= 0.1  # Getting farther
        self.prev_distance = current_distance
        
        # Innovation 2: Loop detection and no-progress penalty
        if new_head in self.position_history:
            reward -= 1.0  # Penalty for revisiting recent positions
            info['loop_detected'] = True
        
        self.position_history.append(new_head)
        
        # Innovation 2: No progress penalty
        if self.steps_without_food > 100:
            reward -= 1.0
            done = True
            info['reason'] = 'no_progress'
        
        # Check max steps
        if self.steps >= self.max_steps:
            done = True
            info['reason'] = 'max_steps'
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _update_direction(self, action: int):
        """Update direction based on action (left, straight, right)"""
        if action == self.ACTION_LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == self.ACTION_RIGHT:
            self.direction = (self.direction + 1) % 4
        # ACTION_STRAIGHT: direction stays the same
    
    def _place_food(self):
        """Place food at random empty location"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))
        
        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            # Snake filled the entire grid (unlikely but possible)
            self.food = None
    
    def _get_distance_to_food(self) -> float:
        """Calculate Manhattan distance to food"""
        if self.food is None:
            return 0.0
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation
        
        State vector (9 dimensions):
        - Relative food x (normalized)
        - Relative food y (normalized)
        - Danger front
        - Danger left
        - Danger right
        - Direction one-hot (4 values)
        """
        head_x, head_y = self.snake[0]
        
        # Relative food position (normalized to [-1, 1])
        if self.food:
            food_x, food_y = self.food
            rel_food_x = (food_x - head_x) / self.grid_size
            rel_food_y = (food_y - head_y) / self.grid_size
        else:
            rel_food_x = 0.0
            rel_food_y = 0.0
        
        # Danger detection
        danger_front = self._is_danger(self.direction)
        danger_left = self._is_danger((self.direction - 1) % 4)
        danger_right = self._is_danger((self.direction + 1) % 4)
        
        # Direction one-hot encoding
        dir_one_hot = [0, 0, 0, 0]
        dir_one_hot[self.direction] = 1
        
        state = np.array([
            rel_food_x,
            rel_food_y,
            float(danger_front),
            float(danger_left),
            float(danger_right),
            *dir_one_hot
        ], dtype=np.float32)
        
        return state
    
    def _is_danger(self, direction: int) -> bool:
        """Check if there's danger (wall or snake body) in given direction"""
        head_x, head_y = self.snake[0]
        
        # Calculate next position in given direction
        if direction == self.DIR_UP:
            next_pos = (head_x, head_y - 1)
        elif direction == self.DIR_DOWN:
            next_pos = (head_x, head_y + 1)
        elif direction == self.DIR_LEFT:
            next_pos = (head_x - 1, head_y)
        else:  # RIGHT
            next_pos = (head_x + 1, head_y)
        
        # Check wall collision
        if (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
            next_pos[1] < 0 or next_pos[1] >= self.grid_size):
            return True
        
        # Check self collision
        if next_pos in self.snake[:-1]:  # Exclude tail as it will move
            return True
        
        return False
    
    def render(self):
        """Print text representation of the game (for debugging)"""
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = 'H'  # Head
            else:
                grid[y][x] = 'S'  # Body
        
        # Place food
        if self.food:
            food_x, food_y = self.food
            grid[food_y][food_x] = 'F'
        
        # Print grid
        print('=' * (self.grid_size * 2 + 2))
        for row in grid:
            print('|' + ''.join(f'{cell} ' for cell in row) + '|')
        print('=' * (self.grid_size * 2 + 2))
        print(f'Score: {self.score} | Steps: {self.steps}')


