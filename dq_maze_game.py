import maze_generator
import numpy as np
import random
import torch.nn as nn
from collections import deque

'''
game env class for deep q learning, 7x7 mazes, as well as classes for
q networks, and a replay buffer.
'''

class ReplayBuffer:
    '''replay buffer class'''
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
class QNetwork(nn.Module):
    '''QNetwork class, used for dq no doors agent'''
    def __init__(self):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(49, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)

class QNetworkDoors(nn.Module):
    '''Qnetwork class, for dq doors agent'''
    def __init__(self):
        super(QNetworkDoors, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.net(x)

class DQ_MazeGame:
    '''game env'''
    def __init__(self, is_doors):
        if is_doors:
            self.orig_maze= maze_generator.generate_maze_7(True) # doors
        else:    
            self.orig_maze = maze_generator.generate_maze_7(False) # no doors
        self.maze = (self.orig_maze.copy())
        self.exit_pos = np.argwhere(self.maze == 2) # get exit position
        self.agent_pos = 1, 1 # set agent position
        self.current_tile = 0 # keep track of tile that player is on
        self.adjacent = self.get_adjacent() # adjacent tiles
        self.door_adjacent = self.get_door_adj() # boolean if door is adjacent
        self.on_key = self.is_on_key() # bool whether agent on key
        self. key_state = 0 # 0: no keys, 1: red key, 2: yellow key, 3: both
        self.path = set((1, 1)) # keep track of traversed tiles
        
        # actions
        self.actions = ['up', 'down', 'left', 'right', 'open', 'pickup']
        
        # rewards
        self.rewards = {
            'exit': 1000,
            'move': 0,
            'retrace': 0,
            'pickup_key': 0,
            'no_key': -50,
            'open_door': 0,
            'no_door': -50,
            'hit_wall': -100
        }
    
    def get_adjacent(self):
        '''function gets adjacent tile values'''
        x, y = self.agent_pos
        above = int(self.maze[x - 1, y])
        below = int(self.maze[x + 1, y])
        left = int(self.maze[x, y - 1])
        right = int(self.maze[x, y + 1])
        return above, below, left, right
    
    def get_observation(self):
        '''gets observation, no doors'''
        maze = np.copy(self.maze)
        return maze
    
    def get_observation_doors(self):
        '''gets observation, doors'''
        maze = np.copy(self.maze)
        door_adj = self.door_adjacent
        key_state = self.key_state
        on_key = self.on_key
        return maze, door_adj, key_state, on_key
    
    def random_maze(self, is_doors):
        '''sets a random maze'''
        if is_doors:
            self.maze = maze_generator.generate_maze_7(True)
        else:
            self.maze = maze_generator.generate_maze_7(False)
        self.exit_pos = np.where(self.maze == 2)
        self.agent_pos = 1, 1
        self.current_tile = 0
        self.key_state = 0
        self.door_adjacent = self.get_door_adj()
        self.on_key = False
        x, y = self.exit_pos
        self.maze[x, y] = 2
        self.path.clear()
        self.path.add((1, 1))
        
    def is_on_key(self):
        '''function returns true if on key, else false'''
        if self.current_tile in {4, 6}:
            return True
        else:
            return False
        
    def pickup_key(self):
        '''function picks up key for agent, sets keystate'''
        pickup = self.rewards['pickup_key']
        no_key = self.rewards['no_key']
        
        if self.current_tile == 4:
            self.current_tile = 0
            if self.key_state == 0:
                self.key_state = 1
                return pickup
            else:
                self.key_state = 3
                return pickup
            
        elif self.current_tile == 6:
            self.current_tile = 0
            if self.key_state == 0:
                self.key_state = 2
                return pickup
            else:
                self.key_state = 3
                return pickup
            
        return no_key
    
    def get_door_adj(self):
        '''returns true if door adjacent, else false'''
        above, below, left, right = self.get_adjacent()
        if above in {5, 7} \
        or below in {5, 7} \
        or left in {5, 7} \
        or right in {5, 7}:
            return True
        return False
                
    def open_door(self):
        '''function opens door for agent'''
        x, y = self.agent_pos
        above, below, left, right = self.get_adjacent()
        no_door = self.rewards['no_door']
        door_opened = self.rewards['open_door']
        
        if not self.door_adjacent:
            return no_door
        
        if above in {5, 7}:
            if above == 5:
                if self.key_state in {1, 3}:
                    self.maze[x - 1, y] = 0
                    return door_opened
            else:
                if self.key_state in {2, 3}:
                    self.maze[x - 1, y] = 0
                    return door_opened
                
        elif below in {5, 7}:
            if below == 5:
                if self.key_state in {1, 3}:
                    self.maze[x + 1, y] = 0
                    return door_opened
            else:
                if self.key_state in {2, 3}:
                    self.maze[x + 1, y] = 0
                    return door_opened
                
        elif left in {5, 7}:
            if left == 5:
                if self.key_state in {1, 3}:
                    self.maze[x, y - 1] = 0
                    return door_opened
            else:
                if self.key_state in {2, 3}:
                    self.maze[x, y - 1] = 0
                    return door_opened
                
        elif right in {5, 7}:
            if right == 5:
                if self.key_state in {1, 3}:
                    self.maze[x, y + 1] = 0
                    return door_opened
            else:
                if self.key_state in {2, 3}:
                    self.maze[x, y + 1] = 0
                    return door_opened
                
        return no_door
            
        
    def move_agent(self, direction):
        '''function moves agent'''
        x, y = self.agent_pos
        # penalty of walking on already-traversed tiles
        repeat = self.rewards['retrace']
        # penalty of trying to walk through wall
        hit_wall = self.rewards['hit_wall']
        
        above, below, left, right = self.get_adjacent()
        
        match direction:
            case 'up':
                if above not in {1, 5, 7}:
                    self.agent_pos = x - 1, y
                    self.maze[x, y] = self.current_tile
                    self.current_tile = self.maze[x - 1, y]
                    self.maze[x - 1, y] = 3
                    self.adjacent = self.get_adjacent()
                    if (x - 1, y) in self.path:
                        return repeat
                    else:
                        self.path.add((x - 1, y))
                        return 0
                else:
                    return hit_wall
                
            case 'down':
                if below not in {1, 5, 7}:
                    self.agent_pos = x + 1, y
                    self.maze[x, y] = self.current_tile
                    self.current_tile = self.maze[x + 1, y]
                    self.maze[x + 1, y] = 3
                    self.adjacent = self.get_adjacent()
                    if (x + 1, y) in self.path:
                        return repeat
                    else:
                        self.path.add((x + 1, y))
                        return 0
                else:
                    return hit_wall
                
            case 'left':
                if left not in {1, 5, 7}:
                    self.agent_pos = x, y - 1
                    self.maze[x, y] = self.current_tile
                    self.current_tile = self.maze[x, y - 1]
                    self.maze[x, y - 1] = 3
                    self.adjacent = self.get_adjacent()
                    if (x, y - 1) in self.path:
                        return repeat
                    else:
                        self.path.add((x, y - 1))
                        return 0
                else:
                    return hit_wall
                
            case 'right':
                if right not in {1, 5, 7}:
                    self.agent_pos = x, y + 1
                    self.maze[x, y] = self.current_tile
                    self.current_tile = self.maze[x, y + 1]
                    self.maze[x, y + 1] = 3
                    self.adjacent = self.get_adjacent()
                    if (x, y + 1) in self.path:
                        return repeat
                    else:
                        self.path.add((x, y + 1))
                        return 0
                    
                else:
                    return hit_wall
                
    def set_maze(self, maze):
        '''sets game env maze'''
        self.maze = maze.copy()
        self.orig_maze = maze.copy()
        self.key_state = 0
        self.door_adjacent = self.get_door_adj()
        self.agent_pos = 1, 1
        self.on_key = False
        self.exit_pos = np.where(maze == 2)
        self.path.clear()
        self.path.add((1, 1))
        self.current_tile = 0
        
    def step(self, action):
        '''step for no doors'''
        action = self.actions[action]
        reward = self.move_agent(action)
        maze = self.get_observation()
        done = False
        
        if self.current_tile == 2:
            reward += 1000
            done = True
        
        return maze, reward, done
        
    def step_doors(self, action):
        '''step for doors'''
        action = self.actions[action]
        if action in {'up', 'down', 'left', 'right'}:
            reward = self.move_agent(action)
        elif action == 'open':
            reward = self.open_door()
        elif action == 'pickup':
            reward = self.pickup_key()
            
        self.adjacent = self.get_adjacent()
        self.door_adjacent = self.get_door_adj()
        self.on_key = self.is_on_key()
        done = False
        if self.current_tile == 2:
            reward += 1000
            done = True
        maze, door_adj, key_state, on_key = self.get_observation_doors()
        
        return maze, door_adj, key_state, on_key, reward, done