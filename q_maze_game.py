import maze_generator
import numpy as np

'''
class for q learning game env
'''

class Q_MazeGame:
    '''game env class for q learning'''
    def __init__(self, is_doors):
        if is_doors:
            self.orig_maze = maze_generator.generate_maze_13(True) # generates 13x13 maze with doors
        else:
            self.orig_maze = maze_generator.generate_maze_13(False) # generates 13x13 maze with no doors
        self.exit_pos = np.where(self.orig_maze == 3) # get exit position
        self.maze = (self.orig_maze.copy())
        self.agent_pos = 1, 1 # set agent position
        self.current_tile = 0 # keep track of tile that player is on
        self.adjacent = self.get_adjacent() # get adjacent tiles
        self.on_key = False # keeps track of if agent on key
        self.door_adjacent = self.get_door_adj() # bool whether door is adjacent
        self.key_state = 0 # 0: no keys, 1: red key, 2: yellow key, 3: both
        self.path = set((1, 1)) # keep track of traversed tiles
        
        # actions
        self.actions = ['up', 'down', 'left', 'right', 'open', 'pickup']
        
        self.reward = {
            'exit': 1000,
            'pickup_key': 0,
            'no_key': -100,
            'open_door': 0,
            'no_door': -100,
            'retrace': 0,
            'hit_wall': -100
        }

    def is_on_key(self):
        '''checks if agent is on key'''
        if self.current_tile in {4, 6}:
            return True
        else:
            return False
    
    def get_adjacent(self):
        '''gets adjacent tile values'''
        x, y = self.agent_pos
        above = int(self.maze[x - 1, y])
        below = int(self.maze[x + 1, y])
        left = int(self.maze[x, y - 1])
        right = int(self.maze[x, y + 1])
        return above, below, left, right
    
    def get_observation(self):
        '''gets observation'''
        x, y = self.agent_pos
        door_adj = self.door_adjacent
        key_state = self.key_state
        on_key = self.on_key
        return {'pos': (x, y), 'on_key': on_key, 'door_adj': door_adj, 'key_state': key_state}
    
    def reset_maze(self):
        '''resets maze to original state'''
        self.maze = self.orig_maze.copy()
        self.agent_pos = 1, 1
        self.current_tile = 0
        x, y = self.exit_pos
        self.maze[1, 1] = 3
        self.maze[x, y] = 2
        self.adjacent = self.get_adjacent()
        self.key_state = 0
        self.on_key = False
        self.path.clear()
        self.path.add((1, 1))
        
    def pickup_key(self):
        '''picks up key for agent'''
        pickup = self.reward['pickup_key']
        no_key = self.reward['no_key']
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
        '''opens door'''
        x, y = self.agent_pos
        above, below, left, right = self.get_adjacent()
        no_door = self.reward['no_door']
        door_opened = self.reward['open_door']
        
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
        '''move agent'''
        x, y = self.agent_pos
        # penalty of walking on already-traversed tiles
        repeat = self.reward['retrace']
        # penalty of trying to walk through wall
        hit_wall = self.reward['hit_wall']
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
        self.agent_pos = 1, 1
        exit_pos = np.where(maze == 3)
        self.exit_pos = exit_pos
        self.key_state = 0
        self.on_key = False
        self.door_adjacent = self.get_door_adj
        self.path.clear()
        self.path.add((1, 1))
        self.current_tile = 0
        
    def step(self, action):
        '''step'''
        action = self.actions[action]
        if action in {'up', 'down', 'left', 'right'}:
            reward = self.move_agent(action)
        elif action == 'open':
            reward = self.open_door()
        elif action == 'pickup':
            reward = self.pickup_key()
        obs = self.get_observation()
        self.adjacent = self.get_adjacent()
        self.door_adjacent = self.get_door_adj()
        self.on_key = self.is_on_key()
        done = False
        if self.current_tile == 2:
            reward += 1000
            done = True
        obs = self.get_observation()
        
        return obs, reward, done