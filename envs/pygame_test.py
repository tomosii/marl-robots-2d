import pygame
import sys
from pygame import gfxdraw

pygame.init()

WIDTH = 800
HEIGHT = 800
AGENT_SIZE = 50

BG_COLOR = (16, 16, 16)
AGENT_BLUE = (97, 205, 205)

FPS = 60

ACC = 0.5
FRIC = -0.12

screen = pygame.display.set_mode([700, 700])
pygame.display.set_caption("Pygame Test")
clock = pygame.time.Clock()


class Agent(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([AGENT_SIZE, AGENT_SIZE], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, AGENT_BLUE,
                           (AGENT_SIZE // 2, AGENT_SIZE // 2), AGENT_SIZE // 2)

        self.pos = pygame.math.Vector2(350, 500)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)

    def move(self):
        self.acc = pygame.math.Vector2(0, 0)

        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[pygame.K_LEFT]:
            self.acc.x = -ACC
        if pressed_keys[pygame.K_RIGHT]:
            self.acc.x = ACC
        if pressed_keys[pygame.K_UP]:
            self.acc.y = -ACC
        if pressed_keys[pygame.K_DOWN]:
            self.acc.y = ACC

        self.acc += self.vel * FRIC
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        if self.pos.y > HEIGHT - AGENT_SIZE // 2:
            self.pos.y = HEIGHT - AGENT_SIZE // 2
            self.vel.y = 0
        if self.pos.y < AGENT_SIZE // 2:
            self.pos.y = AGENT_SIZE // 2
            self.vel.y = 0
        if self.pos.x > WIDTH - AGENT_SIZE // 2:
            self.pos.x = WIDTH - AGENT_SIZE // 2
            self.vel.x = 0
        if self.pos.x < AGENT_SIZE // 2:
            self.pos.x = AGENT_SIZE // 2
            self.vel.x = 0

        self.rect.center = self.pos


agent1 = Agent()
all_sprites = pygame.sprite.Group(agent1)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.fill(BG_COLOR)

    agent1.move()
    all_sprites.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
