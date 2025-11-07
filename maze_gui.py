import numpy as np
import pygame as pg
import user_control
import q_no_doors
import q_doors
import maze_generator
import game_data
import dq_no_doors
import dq_doors

# constants
TILE_SIZE = game_data.TILE_SIZE
COLORS = game_data.COLORS
BAR_HEIGHT = game_data.BAR_HEIGHT
Q_MAZE_SIZE = game_data.Q_MAZE_SIZE
DQ_MAZE_SIZE = game_data.DQ_MAZE_SIZE

def create_grid(is_13):
    '''function takes boolean (True = 13x13, False = 7x7) and creates grid'''
    # set grid and font size
    if is_13:
        grid_size = game_data.Q_MAZE_SIZE
        font_size = 20
        instructions = "W: No doors, T: train agent | E: Doors, Y: train agent | R: User control"
    else:
        grid_size = game_data.DQ_MAZE_SIZE
        font_size = 10
        instructions = "W: No doors, T: test agent | E: Doors, Y: test agent | R: User control"
        
    mode = 0 # determines state user is in and allows/blocks certain actions so app doesn't break for 7x7

    # initialize Pygame
    pg.init()

    # calculate maze size
    maze_size = grid_size * TILE_SIZE

    # set font
    font = pg.font.SysFont("Arial", font_size)


    # create grid matrix, add agent
    grid_array = np.ones((grid_size, grid_size))
    if is_13:
        grid_array[1, 1] = 3

    # set diisplay mode
    screen = pg.display.set_mode((maze_size, maze_size + BAR_HEIGHT))
    pg.display.set_caption("Maze Editor")

    # main loop
    running = True
    while running:
        screen.fill(COLORS[0])
        # draw maze
        for row in range(grid_size):
            for col in range(grid_size):
                rect = pg.Rect(col * TILE_SIZE, row * TILE_SIZE + BAR_HEIGHT, TILE_SIZE, TILE_SIZE)
                tile_val = grid_array[row][col]
                tile_color = COLORS[int(tile_val)]
                # circle for keys
                if tile_val in {4, 6}:
                    pg.draw.circle(screen, tile_color, rect.center, TILE_SIZE // 2)
                else:
                    pg.draw.rect(screen, tile_color, rect)

        # grid lines for edit/generate mode
        for row in range(grid_size + 1):
            y = row * TILE_SIZE + BAR_HEIGHT
            pg.draw.line(screen, (0, 0, 0), (0, y), (maze_size, y))
        for col in range(grid_size + 1):
            x = col * TILE_SIZE
            pg.draw.line(screen, (0, 0, 0), (x, BAR_HEIGHT), (x, maze_size + BAR_HEIGHT))

        # create bar with instructions
        pg.draw.rect(screen, (50, 50, 50), (0, 0, maze_size, BAR_HEIGHT))
        text_surface = font.render(instructions, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(maze_size // 2, BAR_HEIGHT // 2))
        screen.blit(text_surface, text_rect)

        # handle events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            
            # check if 13x13, q learning maze
            elif is_13:
                pg.display.set_caption("Generating/Editing Maze")
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False # ESC: quit/back
                    if event.key == pg.K_w:
                        rand_maze = maze_generator.generate_maze_13(False) # W: generate no doors maze
                        grid_array = rand_maze.copy()
                    if event.key == pg.K_e:
                        rand_maze = maze_generator.generate_maze_13(True) # E: generate doors maze
                        grid_array = rand_maze.copy()
                    if event.key == pg.K_r:
                        # update instructions bar and run user controls
                        old_instructions = instructions
                        instructions = "WASD or arrows: Move | Spacebar: Interact | ESC: Back"
                        user_control.run(grid_array, Q_MAZE_SIZE, screen, instructions, font) # R: user control
                        instructions = old_instructions
                    if event.key == pg.K_t:
                        q_no_doors.run_custom(grid_array, screen) # T: train/run no doors agent
                    if event.key == pg.K_y:
                        q_doors.run_custom(grid_array, screen) # Y: train/run doors agent
            else: # if 7x7
                pg.display.set_caption("Generating Maze")
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False # Q: quit/back
                    if event.key == pg.K_w:
                        rand_maze = maze_generator.generate_maze_7(False) # W: generate no doors maze
                        grid_array = rand_maze.copy()
                        mode = 1
                    if event.key == pg.K_e:
                        rand_maze = maze_generator.generate_maze_7(True) # generate doors maze
                        grid_array = rand_maze.copy()
                        mode = 2
                    if event.key == pg.K_r:
                        # update instructions bar and run user controls
                        old_instructions = instructions
                        instructions = "WASD or arrows: Move | Spacebar: Interact | ESC: Back"
                        user_control.run(grid_array, DQ_MAZE_SIZE, screen, instructions, font) # R: user control
                        instructions = old_instructions
                    if event.key == pg.K_t:
                        if mode == 1:
                            dq_no_doors.run_custom(grid_array, screen) # T: run no doors agent
                    if event.key == pg.K_y:
                        if mode == 2:
                            dq_doors.run_custom(grid_array, screen) # Y: run doors agent

            # check if 13x13
            if is_13:
                # handle mouse clicks to modify tile types
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    adjusted_y = mouse_y - BAR_HEIGHT
                    col = mouse_x // TILE_SIZE
                    row = adjusted_y // TILE_SIZE

                    # ensure the click is within the grid area and not on the outer edge and not agent tile
                    if (1 <= row < grid_size - 1 and 1 <= col < grid_size - 1 and not (row == 1 and col == 1)):
                        current_color = grid_array[row][col]
                        current_index = None
                        for key in COLORS:
                            if key == int(current_color):
                                current_index = key
                                break
                        
                        # cycle through tile types with left and right click
                        if event.button == pg.BUTTON_LEFT:
                            if current_index is not None:
                                next_index = (current_index - 1) % len(COLORS)
                                while next_index == 3:
                                    next_index = (next_index - 1) % len(COLORS)
                                grid_array[row][col] = next_index
                        elif event.button == pg.BUTTON_RIGHT:
                            if current_index is not None:
                                next_index = (current_index + 1) % len(COLORS)
                                while next_index == 3:
                                    next_index = (next_index + 1) % len(COLORS)
                                grid_array[row][col] = next_index

        pg.display.flip()

    pg.quit()
