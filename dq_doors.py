import pygame as pg
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import dq_maze_game
import game_data

'''
functions for training, running, and testing deep q learning agents
for mazes with doors and keys.
'''

# constants
TILE_SIZE = game_data.TILE_SIZE
COLORS = game_data.COLORS
BAR_HEIGHT = game_data.BAR_HEIGHT
MAZE_SIZE = game_data.DQ_MAZE_SIZE

def encode_state(maze, door_adj, key_state, on_key):
    '''function takes state info and creates encoded state'''
    state = np.copy(maze)
    state = state.flatten()
    state = np.concatenate([state, [door_adj, key_state, on_key]])

    return torch.tensor(state, dtype=torch.float32)
     
def DQ_learning(num_episodes=1000, gamma=0.9, epsilon=1, decay=0.995, min_epsilon=0.1):
    '''function trains a deep q learning agent on a maze with doors/keys'''
    # initialize game env, q network, target q network
    game = dq_maze_game.DQ_MazeGame(True)
    q_net = dq_maze_game.QNetworkDoors()
    target_net = dq_maze_game.QNetworkDoors()
    target_net.load_state_dict(q_net.state_dict())
    
    # initialize optimizer and replay buffer
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = dq_maze_game.ReplayBuffer()
    batch_size = 64
    update_target_every = 200
    
    # list that keeps track of losses
    losses = []
    
    steps = 0 # keep track of total global steps across episodes
	# run each episode
    for ep in range(num_episodes):
        
        if ep % 50 == 0:
            print(ep)

        game.random_maze(True) # create random maze (True = doors, False = no doors)
        maze, door_adj, key_state, on_key = game.get_observation_doors()
        state = encode_state(maze, door_adj, key_state, on_key)
        
        done = False
        loss_total = 0 # keep track of total loss
        loss_count = 0 # how many times we get/store the loss
        
        # max episode steps, and ep_steps to keep track of episode steps
        ep_steps = 0
        max_ep_steps = 100000
		# run episode while not terminal state
        while not done and ep_steps < max_ep_steps:
            
            ep_steps += 1
            steps += 1
            
			# roll to explore or exploit
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 5) # explore
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action = q_values.argmax().item() # exploit
                    
			# take action, get new state
            maze, door_adj, key_state, on_key, reward, done = game.step_doors(action)
            new_state = encode_state(maze, door_adj, key_state, on_key)
            # store experience
            buffer.push(state, action, reward, new_state, done)
            state = new_state
            
            # update q network
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, new_states, dones = zip(*batch)
                
                # convert to torch tensors
                states = torch.stack(states)
                new_states = torch.stack(new_states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                dones = torch.tensor(dones, dtype=torch.bool)
                
                # calculate q values
                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                # calculate target q values
                with torch.no_grad():
                    max_next_q = target_net(new_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (~dones)

                # calculate loss
                loss = nn.MSELoss()(q_values, target_q)
                loss_total += loss.item()
                loss_count += 1
                
                # gradient decent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Target network update
            if steps % update_target_every == 0:
                target_net.load_state_dict(q_net.state_dict())
                
        # print average loss every 50 eps
        if ep % 50 == 0 and loss_count > 0:
            loss_avg = loss_total / loss_count
            print(loss_avg)
            losses.append(loss_avg) # add to loss list
            print(epsilon)
            
        # decay epsilon
        epsilon = max(min_epsilon, epsilon * decay)
        
    return q_net, losses

def learn_save():
    '''function does deep q learning and saves the q net and losses to files'''
    q_net, losses = DQ_learning(num_episodes=5000, gamma=0.9, epsilon=1, decay=0.999999, min_epsilon=0.1)
    torch.save(q_net.state_dict(), "dq_agents/doors_agents/dq_doors_gen_agent11.pth")
    with open("dq_agents/door_agents/doors_agents_loss/loss_values_11.pkl", "wb") as f:
        pickle.dump(losses, f)

def run():
    '''function visually runs trained dq agent on random maze'''
    # initialize game env, q net, and loads agent
    game = dq_maze_game.DQ_MazeGame(True)
    model = dq_maze_game.QNetworkDoors()
    model.load_state_dict(torch.load("dq_agents/doors_agents/dq_doors_gen_agent10.pth"))
    model.eval()

    # get random maze with doors (True = doors)
    game.random_maze(True)

    # setup pygame, screen
    pg.init()
    size = MAZE_SIZE * TILE_SIZE
    screen = pg.display.set_mode((size, size))
    
    def draw_maze():
        # function draws maze
        
        # fill screen with white
        screen.fill(COLORS[0])
        # draw each tile
        for y in range(game.maze.shape[0]):
            for x in range(game.maze.shape[1]):
                rect = pg.Rect(y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                tile_color = COLORS[game.maze[x][y]]
                if game.maze[x][y] in {4, 6}:
                    pg.draw.circle(screen, tile_color, rect.center, TILE_SIZE // 2)
                else:
                    pg.draw.rect(screen, tile_color, rect)

    # get obs
    maze, door_adj, key_state, on_key = game.get_observation_doors()
    
    done = False
    total_reward = 0
    
    while not done:
        # draw/update maze
        screen.fill((0, 0, 0))
        draw_maze()
        pg.display.flip()
        pg.time.delay(100)
        
        # agent actions
        state = encode_state(maze, door_adj, key_state, on_key).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        maze, door_adj, key_state, on_key, reward, done = game.step_doors(action)
        total_reward += reward
    print("Total reward:", total_reward)
    
def run_custom(maze, screen):
    '''function takes maze and screen and runs agent on maze'''
    # initialize game env and q network and load agent
    game = dq_maze_game.DQ_MazeGame(True)
    model = dq_maze_game.QNetworkDoors()
    model.load_state_dict(torch.load("dq_agents/doors_agents/dq_doors_gen_agent10.pth"))
    model.eval()
    
    # set maze in game env
    game.set_maze(maze)

    pg.display.set_caption("Running Maze")
    
    def draw_maze():
        # function draws maze
        for y in range(game.maze.shape[0]):
            for x in range(game.maze.shape[1]):
                rect = pg.Rect(y * TILE_SIZE, x * TILE_SIZE + BAR_HEIGHT, TILE_SIZE, TILE_SIZE)
                tile_color = COLORS[game.maze[x][y]]
                if game.maze[x][y] in {4, 6}:
                    pg.draw.rect(screen, COLORS[0], rect)
                    pg.draw.circle(screen, tile_color, rect.center, TILE_SIZE // 2)
                else:
                    pg.draw.rect(screen, tile_color, rect)

    maze, door_adj, key_state, on_key = game.get_observation_doors()
    
    done = False
    max_steps = 50 # max steps agent can take before assumed stuck
    step = 0
    while not done and step < max_steps:
        step += 1
        # update maze
        draw_maze()
        pg.display.flip()
        pg.time.delay(120)
        
        state = encode_state(maze, door_adj, key_state, on_key).unsqueeze(0)
        
        # agent actions
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        maze, door_adj, key_state, on_key, reward, done = game.step_doors(action)
    
def test_visual():
    '''function tests a agent on several mazes with visual'''
    # initialize game env and q network and load agent
    game = dq_maze_game.DQ_MazeGame(True)
    model = dq_maze_game.QNetworkDoors()
    model.load_state_dict(torch.load("dq_agents/doors_agents/dq_doors_gen_agent9.pth"))
    model.eval()
    
    ep = 0
    num_eps = 100
    correct = 0 # keep track of how many the agent solved
    while ep < num_eps:
        # get random maze with doors (True = doors)
        game.random_maze(True)

        # start pygame, screen
        pg.init()
        size = MAZE_SIZE * TILE_SIZE
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

        maze, door_adj, key_state, on_key = game.get_observation_doors()
        
        done = False
        total_reward = 0
        steps = 0
        max_steps = 50
        while not done and steps < max_steps:
            # update maze
            screen.fill((0, 0, 0))
            draw_maze()
            pg.display.flip()
            pg.time.delay(30)
            
            state = encode_state(maze, door_adj, key_state, on_key).unsqueeze(0)
            
            # agent actions
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()
            maze, door_adj, key_state, on_key, reward, done = game.step_doors(action)
            total_reward += reward
            steps += 1
        if done:
            print("correct!")
            correct += 1
        else:
            print("NOT correct!")
        ep += 1
    print(correct/num_eps)

def test():
    '''function tests agent on a number of mazes, prints accuracy'''
    # initialize game env, q network, loads agent
    game = dq_maze_game.DQ_MazeGame(True)
    model = dq_maze_game.QNetworkDoors()
    model.load_state_dict(torch.load("dq_agents/doors_agents/dq_doors_gen_agent10.pth"))
    model.eval()
    
    correct = 0
    ep = 0
    num_eps = 1000
    while ep < num_eps:
        if ep % 50:
            print(ep)
            
        # get random maze with doors (True = doors)
        game.random_maze(True)

        maze, door_adj, key_state, on_key = game.get_observation_doors()
        
        done = False
        max_steps = 30
        steps = 0
        while not done and steps < max_steps:
            # agent actions
            state = encode_state(maze, door_adj, key_state, on_key).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()
            maze, door_adj, key_state, on_key, reward, done = game.step_doors(action)
            steps += 1
        if done:
            print("correct!")
            correct += 1
        else:
            print("NOT correct!")
        ep += 1
    print(correct/num_eps)
    
    
#learn_save()
#run()
#test()
#test_visual()