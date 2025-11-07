import maze_generator
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
for 7x7 mazes with no doors
'''

# constants
TILE_SIZE = game_data.TILE_SIZE
COLORS = game_data.COLORS
BAR_HEIGHT = game_data.BAR_HEIGHT

def encode_state(maze):
    '''function encodes state'''
    state = np.copy(maze)
    return torch.tensor(state, dtype=torch.float32).flatten()
     
def DQ_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay=0.995, min_epsilon=0.1):
    '''function trains an agent with deep q learning'''
    # initialize game env, q net, target q net
    game = dq_maze_game.DQ_MazeGame(False)
    q_net = dq_maze_game.QNetwork()
    target_net = dq_maze_game.QNetwork()
    target_net.load_state_dict(q_net.state_dict())
    
    # initialize optimizer and buffer
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = dq_maze_game.ReplayBuffer()
    batch_size = 64
    
    losses = [] # list to save losses
    update_target_every = 300
    steps = 0
 
	# run each episode
    for ep in range(num_episodes):
        
        if ep % 10 == 0:
            print(ep)
        
        # get random maze no doors (False = no doors)
        game.random_maze(False)
        
        maze = game.get_observation()
        state = encode_state(maze)
        done = False
        
        loss_total = 0
        loss_count = 0
        ep_steps = 0
        max_steps = 100000
        
		# run episode while not terminal state
        while not done and ep_steps < max_steps:
            ep_steps += 1
            steps += 1
			# roll to explore or exploit
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3) # explore
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action = q_values.argmax().item() # exploit
                    
			# take action
            maze, reward, done = game.step(action)
            new_state = encode_state(maze)
            
            # store experience
            buffer.push(state, action, reward, new_state, done)
            state = new_state
            
            # update q network
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, new_states, dones = zip(*batch)
                
                # convert to tensors
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
                
                # gradient descent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # target network update
            if steps % update_target_every == 0:
                target_net.load_state_dict(q_net.state_dict())

        # print avg loss every 10 eps
        if ep % 10 == 0 and loss_count > 0:
            loss_avg = loss_total / loss_count
            print(loss_avg)
            losses.append(loss_avg)
            print(epsilon)
            
        # decay epsilon
        epsilon = max(min_epsilon, epsilon * decay)
        
    return q_net, losses

def learn_save():
    '''train agent and save it, plus losses'''
    q_net, losses = DQ_learning(num_episodes=200, gamma=0.9, epsilon=1, decay=.995, min_epsilon=0.1)
    torch.save(q_net.state_dict(), "dq_agents/no_doors_agents/dq_gen_agent1.pth")
    with open("dq_agents/no_doors_agents/no_doors_agents_loss/dq_loss_values_1.pkl", "wb") as f:
        pickle.dump(losses, f)

def run():
    '''function runs agent on random maze'''
    # initialize game env, q network, load agent
    game = dq_maze_game.DQ_MazeGame(False)
    model = dq_maze_game.QNetwork()
    model.load_state_dict(torch.load("dq_agents/no_doors_agents/dq_gen_agent2.pth"))
    model.eval()
    
    # get random maze no doors (False = no doors)
    maze = maze_generator.generate_maze_7(False)
    game.set_maze(maze)

    # initialize pygame
    pg.init()
    size = 7 * TILE_SIZE
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

    maze = game.get_observation()
    
    done = False
    total_reward = 0
    while not done:
        # update maze
        screen.fill((0, 0, 0))
        draw_maze()
        pg.display.flip()
        pg.time.delay(200)
        
        # agent actions
        state = encode_state(maze).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        maze, reward, done = game.step(action)
        total_reward += reward

    print("Total reward:", total_reward)
    
def run_custom(maze, screen):
    '''function runus agent on maze'''
    # initialize game env, q network, loads agent
    game = dq_maze_game.DQ_MazeGame(False)
    model = dq_maze_game.QNetwork()
    model.load_state_dict(torch.load("dq_agents/no_doors_agents/dq_gen_agent3.pth"))
    model.eval()

    # set game env maze
    game.set_maze(maze)

    pg.display.set_caption("Running Maze")
    
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

    maze = game.get_observation()
    
    done = False
    total_reward = 0
    max_steps = 16
    step = 0
    while not done and step < max_steps:
        # update maze
        draw_maze()
        pg.display.flip()
        pg.time.delay(120)
        
        state = encode_state(maze).unsqueeze(0)
        
        # agent actions
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        maze, reward, done = game.step(action)
        total_reward += reward
        step += 1
        
def test():
    '''tests agent on a number of mazes and prints accuracy'''
    # initialize game env, q network, loads agent
    game = dq_maze_game.DQ_MazeGame(True)
    model = dq_maze_game.QNetwork()
    model.load_state_dict(torch.load("dq_agents/no_doors_agents/dq_gen_agent5.pth"))
    model.eval()
    
    correct = 0
    ep = 0
    num_eps = 1000
    total_reward = 0
    while ep < num_eps:
        if ep % 10:
            print(ep)
        
        # get random maze no doors (False = no doors)
        game.random_maze(False)

        maze = game.get_observation()
        
        done = False
        max_steps = 30
        steps = 0
        while not done and steps < max_steps:
            # agent actions
            state = encode_state(maze).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()
            maze, reward, done = game.step(action)
            
            steps += 1
            
        if done:
            print("correct!")
            correct += 1
            total_reward += reward
        else:
            print("NOT correct!")
        ep += 1
        
    print(correct/num_eps)
    print(total_reward/num_eps)

#learn_save()
#run()
#test()