import numpy as np
import pygame




class drone:
    def __init__(self, position):
        self.position = position
        self.target = np.array([0.0, 0.0])
    
    def update(self):
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction /= distance
            self.position += direction * min(distance, 0.1)

    def draw(self, ax):
        pygame.draw.circle(ax, (255, 0, 0), (int(self.position[0]), int(self.position[1])), 5)
        pygame.draw.line(ax, (0, 255, 0), (int(self.position[0]), int(self.position[1])), (int(self.target[0]), int(self.target[1])), 2)



if __name__ == "__main__":


    drones = [drone(np.array([100, 100])), drone(np.array([200, 200]))]

    pygame.init()
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Drone Simulation")
    clock = pygame.time.Clock()
    ax = pygame.Surface(screen_size)
    ax.fill((255, 255, 255))  # Fill background with white
    running = True

    def loop():
        for d in drones:
            d.update()
            d.draw(ax)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ax.fill((255, 255, 255))

