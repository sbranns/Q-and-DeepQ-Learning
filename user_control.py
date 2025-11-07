import pygame as pg
import game_data

'''
functions related to user control of agent
'''

# constants
TILE_SIZE = game_data.TILE_SIZE
COLORS = game_data.COLORS
BAR_HEIGHT = game_data.BAR_HEIGHT

# keep track of keys
inventory = set()

# keeps track of the tile player is currently on
current_tile = 0

def draw_maze(screen):
    '''draws maze'''
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            rect = pg.Rect(y * TILE_SIZE, x * TILE_SIZE + BAR_HEIGHT, TILE_SIZE, TILE_SIZE)
            tile_color = COLORS[maze[x][y]]
            if maze[x][y] in {4, 6}:
                pg.draw.rect(screen, COLORS[0], rect)
                pg.draw.circle(screen, tile_color, rect.center, TILE_SIZE // 2)
            else:
                pg.draw.rect(screen, tile_color, rect)
            
def move_player(x, y):
    '''moves player'''
    global player_pos
    global current_tile
    
    current_x, current_y = player_pos
    new_x = current_x + x
    new_y = current_y + y
    
    desired_pos = maze[new_x, new_y]
    if desired_pos in {0, 4, 6}:
        maze[new_x, new_y] = 3
        maze[current_x, current_y] = current_tile
        current_tile = desired_pos
        player_pos = new_x, new_y
    elif desired_pos == 2:
        return False
    return True
        
def interact():
    '''pick up key or open door'''
    global current_tile
    x, y = player_pos
    if current_tile in {4, 6}:
        inventory.add(current_tile)
        current_tile = 0
    else:
        if 4 in inventory:
            if maze[x + 1, y] == 5:
                maze[x + 1, y] = 0
            if maze[x - 1, y] == 5:
                maze[x - 1, y] = 0
            if maze[x, y + 1] == 5:
                maze[x, y + 1] = 0
            if maze[x, y - 1] == 5:
                maze[x, y - 1] = 0
        if 6 in inventory:
            if maze[x + 1, y] == 7:
                maze[x + 1, y] = 0
            if maze[x - 1, y] == 7:
                maze[x - 1, y] = 0
            if maze[x, y + 1] == 7:
                maze[x, y + 1] = 0
            if maze[x, y - 1] == 7:
                maze[x, y - 1] = 0
                
def get_player_pos(grid_size):
    '''gets player position'''
    for i in range(grid_size):
        for j in range(grid_size):
            if maze[i][j] == 3:
                return i, j

def run(maze_array, grid_size, screen, instructions, font):
    '''runs loop that allows user control'''
    pg.display.set_caption("Running Maze")
    
    inventory.clear()
    global player_pos
    global maze
    maze_size = grid_size * TILE_SIZE
    maze = maze_array.copy()
    player_pos = get_player_pos(grid_size)
    running = True
    while running:
        draw_maze(screen)
        pg.display.flip()
        
        # update instruction bar
        pg.draw.rect(screen, (50, 50, 50), (0, 0, maze_size, BAR_HEIGHT))
        text_surface = font.render(instructions, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(maze_size // 2, BAR_HEIGHT // 2))
        screen.blit(text_surface, text_rect)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                match event.key:
                    case pg.K_UP | pg.K_w:
                        running = move_player(-1, 0)
                    case pg.K_DOWN | pg.K_s:
                        running = move_player(1, 0)
                    case pg.K_RIGHT | pg.K_d:
                        running = move_player(0, 1)
                    case pg.K_LEFT | pg.K_a:
                        running = move_player(0, -1)
                    case pg.K_SPACE:
                        interact()
                    case pg.K_ESCAPE:
                        running = False
    return

pg.quit()