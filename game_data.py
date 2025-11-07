'''
game data used in different files
'''

TILE_SIZE = 40
Q_MAZE_SIZE = 13
DQ_MAZE_SIZE = 7
BAR_HEIGHT = 40 # text bar above maze

COLORS = {
    0: (255, 255, 255),  # Walkable path
    1: (100, 100, 100),  # Wall
    2: (0, 255, 0),      # Exit
    3: (0, 0, 255),      # Player
    4: (255, 0, 0),      # Red key
    5: (255, 0, 0),      # Red Door
    6: (255, 204, 0),    # Yellow key
    7: (255, 204, 0),    # Yellow Door
}