import tkinter as tk
import threading
import maze_gui

'''
functions for launcher menu
'''

def launch_maze(root, create_grid):
    '''function handles launching create_grid'''
    root.withdraw() # hide tkinter window
    def run_and_return():
        create_grid() # launch pygame, create_grid function
        root.deiconify() # reopen tkinter window when pygame window closes
    threading.Thread(target=run_and_return).start()

def launch():
    '''starts program'''
    # create tkinter launch window
    root = tk.Tk()
    root.title("Maze Launcher")
    root.geometry("300x200")
    root.resizable(False, False)

    label = tk.Label(root, text="Select Q Learning Type:", font=("Arial", 14))
    label.pack(pady=20)

    btn_13 = tk.Button(root, text="Q Learning (13x13 Grid)", font=("Arial", 12),
                       command=lambda: launch_maze(root, lambda: maze_gui.create_grid(True)))
    btn_13.pack(pady=5)

    btn_7 = tk.Button(root, text="Deep Q Learning (7x7 Grid)", font=("Arial", 12),
                      command=lambda: launch_maze(root, lambda: maze_gui.create_grid(False)))
    btn_7.pack(pady=5)

    quit_btn = tk.Button(root, text="Quit", font=("Arial", 12), command=root.destroy)
    quit_btn.pack(pady=5)

    root.mainloop()