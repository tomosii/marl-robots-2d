from enum import Enum
import pygame
import sys
from pygame import gfxdraw

pygame.init()

WIDTH = 850
HEIGHT = 850
OFFSET = 100

AGENT_SIZE = 60
ROOM_SIZE = 260
CORRIDOR_WIDTH = AGENT_SIZE + 40

BG_COLOR = (16, 16, 16)
AGENT_BLUE = (97, 205, 205)
ROOM_COLOR = (70, 70, 70)
CORRIDOR_COLOR = (160, 160, 160)

FPS = 60

ACC = 0.5
FRIC = -0.08
MAX_SPEED = 6

screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Pygame Test")
clock = pygame.time.Clock()


class CorridorOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class CorridorPosition(Enum):
    TOPLEFT = 0
    BOTTOMRIGHT = 1


class Room(pygame.sprite.Sprite):
    def __init__(self, center_x, center_y):
        super().__init__()
        self.image = pygame.Surface([ROOM_SIZE, ROOM_SIZE], pygame.SRCALPHA)

        pygame.draw.rect(self.image, ROOM_COLOR, [
                         0, 0, ROOM_SIZE, ROOM_SIZE], 0, 20)

        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Corridor(pygame.sprite.Sprite):
    def __init__(self, start, end, orientation, position):
        super().__init__()
        if orientation == CorridorOrientation.HORIZONTAL:
            width = end[0] - start[0] - ROOM_SIZE // 2 + CORRIDOR_WIDTH // 2
            height = CORRIDOR_WIDTH
            if position == CorridorPosition.TOPLEFT:
                left = start[0] - CORRIDOR_WIDTH // 2
                top = end[1] - CORRIDOR_WIDTH // 2
            else:
                left = start[0] + ROOM_SIZE // 2
                top = start[1] - CORRIDOR_WIDTH // 2
        else:
            width = CORRIDOR_WIDTH
            height = end[1] - start[1] - ROOM_SIZE // 2 + CORRIDOR_WIDTH // 2
            if position == CorridorPosition.TOPLEFT:
                left = end[0] - CORRIDOR_WIDTH // 2
                top = start[1] - CORRIDOR_WIDTH // 2
            else:
                left = start[0] - CORRIDOR_WIDTH // 2
                top = start[1] + ROOM_SIZE // 2
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)

        self.rect = pygame.draw.rect(self.image, CORRIDOR_COLOR, [
            0, 0, width, height], 0, 0)
        self.rect.left = left
        self.rect.top = top


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

        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)

        # x = v0t + 1/2at^2
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
room1 = Room(OFFSET + ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2 - OFFSET)
room2 = Room(WIDTH - ROOM_SIZE // 2 - OFFSET, OFFSET + ROOM_SIZE // 2)
corridor1 = Corridor(room1.rect.center, room2.rect.center,
                     CorridorOrientation.HORIZONTAL, CorridorPosition.TOPLEFT)
corridor2 = Corridor(room2.rect.center, room1.rect.center,
                     CorridorOrientation.VERTICAL, CorridorPosition.TOPLEFT)
corridor3 = Corridor(room1.rect.center, room2.rect.center,
                     CorridorOrientation.HORIZONTAL, CorridorPosition.BOTTOMRIGHT)
corridor4 = Corridor(room2.rect.center, room1.rect.center,
                     CorridorOrientation.VERTICAL, CorridorPosition.BOTTOMRIGHT)


all_sprites = pygame.sprite.Group(
    room1, room2, corridor1, corridor2, corridor3, corridor4, agent1)

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
