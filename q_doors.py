import maze_generator
import pygame as pg
import numpy as np
import random
import pickle
import q_maze_game
import game_data

'''
functions for training, running, and testing q learning agents for
13x13 mazes with doors/keys.
'''

# constants
TILE_SIZE = game_data.TILE_SIZE
COLORS = game_data.COLORS
BAR_HEIGHT = game_data.BAR_HEIGHT

def hash(obs):
    '''function creates hash number from state info'''
    pos = obs['pos']
    x, y = pos
    if obs['door_adj'] == True:
        door_adj = 1
    else:
        door_adj = 0
    key_state = obs['key_state']
    pos_key_hash = x * 52 + y * 4 + key_state
    total_hash = pos_key_hash * 2 + door_adj
    return total_hash
     
def Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=0.999):
    '''function trains q learning agent on maze with doors'''
    # initialize game env, q table, etc.
    game = q_maze_game.Q_MazeGame(True)
    Q_table = {state: np.zeros(6) for state in range(2704)}
    a = np.zeros((2704, 6))
    visits = np.zeros((2704, 6))
    obs = game.get_observation()
    count = num_episodes
	
	# run each episode
    while count > 0:
        obs = game.get_observation()
        steps = 0
		# run episode while not terminal state
        while not done:
            state = int(hash(obs))
			# roll to explore or exploit
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 5)
			# exploit
            else:
                action = np.argmax(Q_table[state])
			# take action
            new_obs, reward, done = game.step(action)
			# get new state hash
            new_state = int(hash(new_obs))

            steps += 1
			# calculate alpha
            a[state][action] = 1 / (1 + visits[state][action])
            visits[state][action] += 1
			# update Q
            Q_table[state][action] = Q_table[state][action] + (a[state][action] * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state][action]))

            obs = new_obs

        # reset maze to original state
        obs = game.reset_maze()
        count -= 1
        epsilon *= decay_rate

    return Q_table, game.maze

def learn_save():
    '''function trains agent, saves q table and maze'''
    Q_table, maze = Q_learning(num_episodes=5000, gamma=0.9, epsilon=1, decay_rate=0.9996)
    
    with open('saved_mazes_and_Q_tables/Q_table_doors.pickle', 'wb') as handle:
       pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    with open('saved_mazes_and_Q_tables/Maze_doors.pickle', 'wb') as handle:
       pickle.dump(maze, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run():
    '''loads agent and maze and runs'''
    # initialize game env
    game = q_maze_game.Q_MazeGame(True)

    # initialize pygame
    pg.init()
    size = 13 * TILE_SIZE
    screen = pg.display.set_mode((size, size))
    
    def draw_maze():
        '''function draws maze'''
        screen.fill(COLORS[0])
        for y in range(game.maze.shape[0]):
            for x in range(game.maze.shape[1]):
                rect = pg.Rect(y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                tile_color = COLORS[game.maze[x][y]]
                if game.maze[x][y] in {4, 6}:
                    pg.draw.circle(screen, tile_color, rect.center, TILE_SIZE // 2)
                else:
                    pg.draw.rect(screen, tile_color, rect)

    # load q table and maze
    Q_table = np.load('saved_mazes_and_Q_tables/Q_table_doors.pickle', allow_pickle=True)
    maze = np.load('saved_mazes_and_Q_tables/Maze_doors.pickle', allow_pickle=True)

    # set maze
    game.set_maze(maze)
    
    obs = game.get_observation()
    done = False
    total_reward = 0
    while not done:
        # update maze
        screen.fill((0, 0, 0))
        draw_maze()
        pg.display.flip()
        pg.time.delay(200)
        
        # agent actions
        state = int(hash(obs))
        action = np.argmax(Q_table[state])
        obs, reward, done = game.step(action)
        total_reward += reward

def Q_learning_custom(game, num_episodes=5000, gamma=0.9, epsilon=1, decay_rate=0.9996, test=False):
    '''function trains agent on specific maze'''
    # create blank q table
    Q_table = {state: np.zeros(6) for state in range(2704)}
    a = np.zeros((2704, 6))
    visits = np.zeros((2704, 6))
    obs = game.get_observation()
    count = num_episodes
	
	# run each episode
    while count > 0:
        # allow user to exit training
        if not test:
            for event in pg.event.get():
                if event.type == pg.QUIT or \
                    event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return False
                
        obs = game.get_observation()
        done = False
		# run episode while not terminal state
        while not done:
            # allow user to exit training
            if not test:
                for event in pg.event.get():
                    if event.type == pg.QUIT or \
                        event.type == pg.KEYDOWN:
                        if event.key == pg.K_ESCAPE:
                            return False
            
            state = int(hash(obs))
            
			# roll to explore or exploit
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 5)
			# exploit
            else:
                action = np.argmax(Q_table[state])
                
			# take action
            new_obs, reward, done = game.step(action)
            
			# get new state hash
            new_state = int(hash(new_obs))
            
			# calculate alpha
            a[state][action] = 1 / (1 + visits[state][action])
            visits[state][action] += 1
            
			# update Q 
            Q_table[state][action] = Q_table[state][action] + (a[state][action] * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state][action]))

            obs = new_obs

        # reset maze
        obs = game.reset_maze()
        count -= 1
        epsilon *= decay_rate

    return Q_table
    
def run_custom(maze, screen):
    '''function takes maze and runs agent on it'''
    # initialize game env
    game = q_maze_game.Q_MazeGame(True)
    # set maze
    game.set_maze(maze)
    
    # initialize pygame, create "training..." text
    pg.font.init()
    font = pg.font.SysFont('Arial', 24)
    text_surface = font.render(f"Training...", True, (255, 255, 255))
    screen.blit(text_surface, (20, 50))  # Top-left corner
    pg.display.flip()
    
    # train agent
    Q_table = Q_learning_custom(game, num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.9997, test=True)
    # exit if training incomplete (if it returned False)
    if not Q_table:
        return

    pg.display.set_caption("Running maze")

    def draw_maze():
        '''function draws maze'''
        for y in range(game.maze.shape[0]):
            for x in range(game.maze.shape[1]):
                rect = pg.Rect(y * TILE_SIZE, x * TILE_SIZE + BAR_HEIGHT, TILE_SIZE, TILE_SIZE)
                tile_color = COLORS[game.maze[x][y]]
                if game.maze[x][y] in {4, 6}:
                    pg.draw.rect(screen, COLORS[0], rect)
                    pg.draw.circle(screen, tile_color, rect.center, TILE_SIZE // 2)
                else:
                    pg.draw.rect(screen, tile_color, rect)

    obs = game.get_observation()
    done = False
    total_reward = 0
    
    while not done:
        # update maze
        draw_maze()
        pg.display.flip()
        pg.time.delay(90)
        
        # agent actions
        state = int(hash(obs))
        action = np.argmax(Q_table[state])
        obs, reward, done = game.step(action)
        total_reward += reward
    
def test():
    '''trains several agents on mazes and tests accuracy'''
    # initialize game env
    game = q_maze_game.Q_MazeGame(True)
    correct = 0
    ep = 0
    num_eps = 100
    total_reward = 0
    while ep < num_eps:
        # generate maze with doors (True = doors)
        maze = maze_generator.generate_maze_13(True)
        # set maze
        game.set_maze(maze)
        # learning
        Q_table = Q_learning_custom(game, num_episodes=500, gamma=0.9, epsilon=1, decay_rate=0.9955, test=True)
        
        if ep % 1 == 0:
            print(ep)
            
        obs = game.get_observation()
        done = False
        max_steps = 100 # steps agent can take before assumed incorrect
        steps = 0
        while not done and steps < max_steps:
            # take action
            state = int(hash(obs))
            action = np.argmax(Q_table[state])
            obs, reward, done = game.step(action)
            total_reward += reward
            steps += 1
        if done:
            print("correct!")
            correct += 1
            total_reward += reward
        else:
            print("NOT correct!")
        ep += 1
    print(correct/num_eps)

#test()
#learn_save()
#run()