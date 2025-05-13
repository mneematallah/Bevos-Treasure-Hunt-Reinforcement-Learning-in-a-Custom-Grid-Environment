
import pygame
pygame.init()
screen = pygame.display.set_mode((512, 512))
pygame.display.set_caption("Test Pygame Window")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255, 255, 255))  
    pygame.display.update()
pygame.quit()