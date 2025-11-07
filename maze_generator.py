import numpy as np
import random
import game_data

'''functions that generate mazes'''

TILE_SIZE = 40

def generate_maze_7(is_doors):
    '''function generates 7x7 maze for dq learning
    is_doors - False: no doors, True: doors'''
    
    grid_size = game_data.DQ_MAZE_SIZE
    last_tile = [(1, 1)] # list for finding the last carved tile at the end
    maze = np.ones((grid_size, grid_size))
    
    # vars for doors/key placements
    global count
    global interval
    count = 1
    if random.randint(1, 2) == 2:
        interval = 1
    else:
        interval = 3
    
    # randomized order of doors and keys
    items = []
    if is_doors:
        rand = random.randint(1, 4)
        if rand == 1:
            items = [4, 6, 5, 7]
        elif rand == 2:
            items = [6, 7, 4, 5]
        elif rand == 3:
            items = [6, 4, 7, 5]
        elif rand == 4:
            items = [4, 5, 6, 7]

    def carve(x, y):
        '''function carves maze paths and places items'''
        global count
        global interval
        
        # choose random direction to carve path
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < grid_size - 1 and 1 <= ny < grid_size - 1 and maze[nx, ny] == 1:
                
                maze[nx - int(dx / 2), ny - int(dy / 2)] = 0
                
                # place items at certain intervals
                if count > interval and items:
                    maze[nx, ny] = items.pop(0)
                    if random.randint(1, 2) == 1:
                        interval += 3
                    else:
                        interval += 2
                else:
                    maze[nx, ny] = 0
                    
                count += 2
                last_tile[0] = nx, ny
                carve(nx, ny) # recurse

    # carve and place agent and exit
    start_x, start_y = 1, 1
    maze[start_x, start_y] = 3
    carve(start_x, start_y)
    maze[last_tile[0]] = 2

    return maze

def generate_maze_13(is_doors):
    '''function generates random 13x13 maze for q learning
    is_doors - False: no doors, True: doors'''
    
    grid_size = game_data.Q_MAZE_SIZE
    last_tile = [(1, 1)]
    maze = np.ones((grid_size, grid_size))
    
    # vars for item placements
    global count
    global interval
    count = 1
    interval = grid_size - 1
    
    # random order of item placements
    items = []
    if is_doors:
        if random.randint(1, 2) == 1:
            items = [4, 6, 5, 7]
        else:
            items = [6, 7, 4, 5]

    def carve(x, y):
        '''function carves maze path and palces items'''
        global count
        global interval
        
        # choose random path direction
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < grid_size - 1 and 1 <= ny < grid_size - 1 and maze[nx, ny] == 1:
                maze[nx - int(dx / 2), ny - int(dy / 2)] = 0
                
                # add items at certain intervals
                if count > interval and items:
                    maze[nx, ny] = items.pop(0)
                    interval += grid_size - 1
                else:
                    maze[nx, ny] = 0
                    
                count += 2
                last_tile[0] = nx, ny
                carve(nx, ny) # recurse

    # carve maze and add agent and exit
    start_x, start_y = 1, 1
    maze[start_x, start_y] = 3
    carve(start_x, start_y)
    maze[last_tile[0]] = 2

    return maze