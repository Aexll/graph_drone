import numpy as np
import pygame
import sys
import os

import threading
import tkinter as tk

DRONE_RADIUS = 10
DRONE_SPEED = 100
CONEXION_RADIUS = 300
TARGET_RADIUS = 5



"""
Note: when a variable start with an underscore it means that the drone is not aware of it, 
it is just for drawing purposes, the drone will not use it to make decisions.
"""

class Drone:
    DRONE_COUNT = 0
    DRONE_REFERENCE = []
    def __init__(self,position, target):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = Drone.DRONE_COUNT
        Drone.DRONE_COUNT += 1
        Drone.DRONE_REFERENCE.append(self)


    # Retarget the drone to a new random position
    def retarget(self):
        self.target = np.random.rand(2) * 500



    def think(self):
        pass



    def get_neighbors(self)->set :
        """
        Returns the list of neighbors of the drone
        """
        ret = set()
        for i in range(Drone.DRONE_COUNT):
            ref = Drone.DRONE_REFERENCE[i]
            if ref is not self and np.linalg.norm(self.position - ref.position) < CONEXION_RADIUS:
                ret.add(ref)
        return ret


    def update(self, dt):
        dir = np.linalg.norm(self.target - self.position)
        if dir > 0:
            dir = (self.target - self.position) / dir
        else:
            dir = np.zeros_like(self.position)
        
        # Dont move if the drone is already at the target
        if np.linalg.norm(self.target - self.position) < TARGET_RADIUS:
            dir = np.zeros_like(dir)
        self.position += dir * DRONE_SPEED * dt

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), self.position.astype(int), DRONE_RADIUS)

        # Draw the target as a cross
        target_pos = self.target.astype(int)
        pygame.draw.line(screen, (0, 255, 0), target_pos - (5, 5), target_pos + (5, 5), 2)
        pygame.draw.line(screen, (0, 255, 0), target_pos - (5, -5), target_pos + (5, -5), 2)

        # draw number of the drone
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.id), True, (255, 255, 255))
        text_rect = text.get_rect(center=self.position.astype(int))
        screen.blit(text, text_rect)

        # Draw lines to neighbors
        for neighbor in self.get_neighbors():
            direction = neighbor.position - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.zeros_like(direction)

            line_start = self.position + direction * DRONE_RADIUS
            line_end = neighbor.position - direction * DRONE_RADIUS

            pygame.draw.line(screen, (255, 0, 0), line_start.astype(int)  , line_end.astype(int), 4)





drones = [
    Drone((100, 100), (200, 200)),
    Drone((300, 300), (400, 400)),
    Drone((500, 500), (600, 600)),
    Drone((700, 100), (800, 200)),
    Drone((200, 500), (300, 600)),
    Drone((400, 100), (500, 200))
]


def N(i):
    """
    returns the set of neighbors of drone i
    """
    return [n.id for n in Drone.DRONE_REFERENCE[i].get_neighbors()]

def ξ(i,j,n):
    """
    returns 1 if drone i is aware of drone j, 0 otherwise
    """
    if n<=0: 
        return 1 if i == j else 0
    
    return max( ξ(l,j,n-1) for l in N(i)+[i] )

inf = 100000000

def ω(i,j,n):
    """
    returns the smallest amount of edges that connects drone i to drone j
    """
    if n <= 0:
        return 0 if i == j else inf
    
    if ξ(i,j,n-1) == ξ(i,j,n): return ω(i,j,n-1)
    else:
        return min(ω(l,j,n-1)  + 1 for l in N(i))
    



TICK_ENABLED = False

def event_handler(event):
    global TICK_ENABLED
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_t:  # Press 't' to toggle TICK_ENABLED
            TICK_ENABLED = not TICK_ENABLED
            print(f"TICK_ENABLED set to {TICK_ENABLED}")
        if event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

def main():
    global TICK_ENABLED
    os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"  # X,Y position for main window
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Simulation")
    clock = pygame.time.Clock()
    tick = 0
    running = True
    while running:
        for event in pygame.event.get():
            event_handler(event)

        screen.fill((0, 0, 0))  # Clear the screen with black

        if tick % 600 == 0:  # Change target every second
            for drone in drones:
                drone.retarget()

        for drone in drones:
            drone.think()

        if TICK_ENABLED:
            for drone in drones:
                drone.update(0.016)

        for drone in drones:
            drone.draw(screen)

        pygame.display.flip()  # Update the display

        tick += 1
        clock.tick(60)
    pygame.quit()
    sys.exit()




# main()

# def run_tkinter():
#     root = tk.Tk()
#     root.title("Tkinter Window")
#     label = tk.Label(root, text="This is the Tkinter window")
#     label.pack()
#     root.mainloop()

def run_tkinter():
    root = tk.Tk()
    root.title("ξ(i, j) Table")

    table_frame = tk.Frame(root)
    table_frame.pack(padx=10, pady=10)

    # Create table headers
    for j in range(Drone.DRONE_COUNT):
        header = tk.Label(table_frame, text=f"j={j}", borderwidth=1, relief="solid", width=6)
        header.grid(row=0, column=j+1)
    for i in range(Drone.DRONE_COUNT):
        header = tk.Label(table_frame, text=f"i={i}", borderwidth=1, relief="solid", width=6)
        header.grid(row=i+1, column=0)

    # Create table cells
    cells = []
    for i in range(Drone.DRONE_COUNT):
        row = []
        for j in range(Drone.DRONE_COUNT):
            lbl = tk.Label(table_frame, text="", borderwidth=1, relief="solid", width=6)
            lbl.grid(row=i+1, column=j+1)
            row.append(lbl)
        cells.append(row)

    def update_table():
        n = Drone.DRONE_COUNT
        for i in range(n):
            for j in range(n):
                value = ξ(i, j, n)
                cells[i][j].config(text=str(value))
        root.after(200, update_table)  # update every 200 ms

    update_table()
    root.mainloop()


def tk_ω_table():
    root = tk.Tk()
    root.title("ω(i, j) Table")

    table_frame = tk.Frame(root)
    table_frame.pack(padx=10, pady=10)

    # Create table headers
    for j in range(Drone.DRONE_COUNT):
        header = tk.Label(table_frame, text=f"j={j}", borderwidth=1, relief="solid", width=6)
        header.grid(row=0, column=j+1)
    for i in range(Drone.DRONE_COUNT):
        header = tk.Label(table_frame, text=f"i={i}", borderwidth=1, relief="solid", width=6)
        header.grid(row=i+1, column=0)

    # Create table cells
    cells = []
    for i in range(Drone.DRONE_COUNT):
        row = []
        for j in range(Drone.DRONE_COUNT):
            lbl = tk.Label(table_frame, text="", borderwidth=1, relief="solid", width=6)
            lbl.grid(row=i+1, column=j+1)
            row.append(lbl)
        cells.append(row)

    def update_table():
        n = Drone.DRONE_COUNT
        for i in range(n):
            for j in range(n):
                value = ω(i, j, n)
                cells[i][j].config(text=str(value) if value < inf else "∞")
        root.after(200, update_table)  # update every 200 ms

    update_table()
    root.mainloop()


if __name__ == "__main__":
    pygame_thread = threading.Thread(target=main)
    pygame_thread.start()
    # run_tkinter()
    tk_ω_table()