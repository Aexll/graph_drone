import threading
import pygame
import tkinter as tk

def run_pygame():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Pygame Window")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 128, 255))
        pygame.display.flip()
    pygame.quit()

def run_tkinter():
    root = tk.Tk()
    root.title("Tkinter Window")
    label = tk.Label(root, text="This is the Tkinter window")
    label.pack()
    button = tk.Button(root, text="Quit", command=root.quit)
    button.pack()
    root.mainloop()

if __name__ == "__main__":
    pygame_thread = threading.Thread(target=run_pygame)
    pygame_thread.start()
    run_tkinter()