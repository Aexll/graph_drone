


"""
def info_screen():
    global TICK_ENABLED
    os.environ['SDL_VIDEO_WINDOW_POS'] = "900,100"  # X,Y position for info window
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption("Info Screen")
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            event_handler(event)
            # if event.type == pygame.QUIT:
            #     running = False
            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_t:  # Press 't' to toggle TICK_ENABLED
            #         TICK_ENABLED = not TICK_ENABLED
            #         print(f"TICK_ENABLED set to {TICK_ENABLED}")



        screen.fill((10, 10, 10))


        # Draw the title and neighbors information
        font = pygame.font.SysFont(None, 24)
        title = font.render("Neighbors", True, (255, 255, 255))
        screen.blit(title, (80, 20))
        for i in range(Drone.DRONE_COUNT):
            text = font.render(f"{i}: {N(i)}", True, (255, 255, 255))
            screen.blit(text, (80, 80+24*i))

        # Draw the tick information
        tick_text = font.render(f"Tick: {pygame.time.get_ticks() // 1000}", True, (255, 255, 255))
        screen.blit(tick_text, (80, 240))

        # Draw the ξ information in form of a table
        ξ_title = font.render("ξ(i,j,n)", True, (255, 255, 255))
        screen.blit(ξ_title, (80, 260))

        # X-axis labels
        x_labels = [f"{i}" for i in range(Drone.DRONE_COUNT)]
        for j, label in enumerate(x_labels):
            label_surface = font.render(label, True, (255, 255, 255))
            screen.blit(label_surface, (80 + 40 * j, 280))
        # Y-axis labels and values
        for i in range(Drone.DRONE_COUNT):
            y_label = font.render(f"{i}", True, (255, 255, 255))
            screen.blit(y_label, (40, 300 + 20 * i))
            for j in range(Drone.DRONE_COUNT):
                value = ξ(i, j, Drone.DRONE_COUNT)
                value_surface = font.render(str(value), True, (255, 255, 255))
                screen.blit(value_surface, (80 + 40 * j, 300 + 20 * i))






        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=main)
    p2 = multiprocessing.Process(target=info_screen)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
"""