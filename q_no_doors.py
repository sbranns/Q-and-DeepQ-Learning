import maze_generator
import pygame as pg
import numpy as np
import random
import pickle
import q_maze_game
import game_data

'''
functions related to training, running, and testing q learning agents for 13x13
mazes with no doors.
'''

# constants
TILE_SIZE = game_data.TILE_SIZE
COLORS = game_data.COLORS
BAR_HEIGHT = game_data.BAR_HEIGHT

def hash(obs):
    '''functions takes obs and creates a hash'''
    pos = obs['pos']
    x, y = pos
    return 13 * x + y
     
def Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=0.999):
    '''trains agent using q learning, no doors'''
    # initialize game env and q table
    game = q_maze_game.Q_MazeGame(False)   
    Q_table = {state: np.zeros(4) for state in range(169)}
    a = np.zeros((169, 4))
    visits = np.zeros((169, 4))
    obs = game.get_observation()
    count = num_episodes
	
	# run each episode
    while count > 0:
        print(count)
        obs = game.get_observation()
        done = False
		# run episode while not terminal state
        while not done:
            state = int(hash(obs))
			# roll to explore or exploit
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
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
            try:
                Q_table[state][action] = Q_table[state][action] + (a[state][action] * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state][action]))
            except:
                print(new_obs)
            obs = new_obs

        obs = game.reset_maze()
        count -= 1
        epsilon *= decay_rate

    return Q_table, game.maze

def learn_save():
    '''train agent and save q table and maze'''
    Q_table, maze = Q_learning(num_episodes=200, gamma=0.9, epsilon=1, decay_rate=.989)

    with open('saved_mazes_and_Q_tables/Q_table.pickle', 'wb') as handle:
       pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    with open('saved_mazes_and_Q_tables/Q_Maze.pickle', 'wb') as handle:
       pickle.dump(maze, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run():
    '''runs agent on maze'''
    # initialize game env
    game = q_maze_game.Q_MazeGame(False)

    # initialize pygame
    pg.init()
    size = 13 * TILE_SIZE
    screen = pg.display.set_mode((size, size))
    
    def draw_maze():
        # function draws maze
        
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
    Q_table = np.load('saved_mazes_and_Q_tables/Q_table.pickle', allow_pickle=True)
    maze = np.load('saved_mazes_and_Q_tables/Maze.pickle', allow_pickle=True)

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
        print(action)
        obs, reward, done = game.step(action)
        total_reward += reward

    print("Total reward:", total_reward)
    
def Q_learning_custom(game, num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=0.999, test=False):
    '''teaches agent on specific maze'''
    # create blank q table
    Q_table = {state: np.zeros(4) for state in range(169)}
    a = np.zeros((169, 4))
    visits = np.zeros((169, 4))
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
                action = random.randint(0, 3)
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
            try:
                Q_table[state][action] = Q_table[state][action] + (a[state][action] * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state][action]))
            except:
                pass
            obs = new_obs

        obs = game.reset_maze()
        count -= 1
        epsilon *= decay_rate

    return Q_table

def run_custom(maze, screen):
    '''run agent on maze'''
    # initialize game env, set maze
    game = q_maze_game.Q_MazeGame(False)
    game.set_maze(maze)
    
    # create "loading..." text
    pg.font.init()
    font = pg.font.SysFont('Arial', 24)
    text_surface = font.render(f"Training...", True, (255, 255, 255))
    screen.blit(text_surface, (20, 50))  # Top-left corner
    pg.display.flip()
    
    # training
    Q_table = Q_learning_custom(game, num_episodes=500, gamma=0.9, epsilon=1, decay_rate=0.9955)
    # exit if q table not complete (returned false)
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
    '''function trains agents on a variety of mazes and tests of accurate they are'''
    # initialize game env
    game = q_maze_game.Q_MazeGame(False)
    correct = 0
    ep = 0
    num_eps = 100
    total_reward = 0
    while ep < num_eps:
        # generate 13x13 maze with no doors (False = no doors)
        maze = maze_generator.generate_maze_13(False)
        # set maze
        game.set_maze(maze)
        # train
        Q_table = Q_learning_custom(game, num_episodes=1000, gamma=0.9, epsilon=1, decay_rate=0.9977, test=True)
        
        if ep % 1 == 0:
            print(ep)
            
        obs = game.get_observation()
        done = False
        max_steps = 100
        steps = 0
        while not done and steps < max_steps:
            # actions
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
# learn_save()
# run()