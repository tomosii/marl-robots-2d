from enum import Enum
from typing import List
import pygame
import sys

# from pygame import gfxdraw

pygame.init()

WIDTH = 700
HEIGHT = 700
OFFSET = 100
INFO_HEIGHT = 200

AGENT_SIZE = 60
ROOM_SIZE = 260
GOAL_SIZE = 50
CORRIDOR_WIDTH = AGENT_SIZE + 40
WALL_WIDTH = 2

BG_COLOR = (16, 16, 16)
AGENT_BLUE = (70, 180, 180)
GOAL_BLUE = (50, 140, 140)
ROOM_COLOR = (70, 70, 70)
CORRIDOR_COLOR = (120, 120, 120)
WALL_COLOR = (200, 200, 200)
INFO_BG_COLOR = (30, 30, 30)

FPS = 60

ACC = 0.5
FRIC = -0.12
MAX_SPEED = 6

screen = pygame.display.set_mode([WIDTH + 2 * OFFSET, HEIGHT + OFFSET + INFO_HEIGHT])
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

        pygame.draw.rect(self.image, ROOM_COLOR, [0, 0, ROOM_SIZE, ROOM_SIZE], 0, 20)

        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Goal(pygame.sprite.Sprite):
    def __init__(self, center_x, center_y, color):
        super().__init__()
        self.image = pygame.Surface([GOAL_SIZE, GOAL_SIZE], pygame.SRCALPHA)

        pygame.draw.rect(self.image, color, [0, 0, GOAL_SIZE, GOAL_SIZE], 0, 5)

        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Wall(pygame.sprite.Sprite):
    def __init__(self, start, length, orientation):
        super().__init__()
        if orientation == CorridorOrientation.HORIZONTAL:
            width = length
            height = WALL_WIDTH
            center_x = start[0] + length / 2
            center_y = start[1]
        else:
            width = WALL_WIDTH
            height = length
            center_x = start[0]
            center_y = start[1] + length / 2
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)
        pygame.draw.rect(self.image, WALL_COLOR, [0, 0, width, height])
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

        self.rect = pygame.draw.rect(
            self.image, CORRIDOR_COLOR, [0, 0, width, height], 0, 0
        )
        self.rect.left = left
        self.rect.top = top


class Agent(pygame.sprite.Sprite):
    def __init__(self, center_x, center_y):
        super().__init__()
        self.image = pygame.Surface([AGENT_SIZE, AGENT_SIZE], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(
            self.image, AGENT_BLUE, (AGENT_SIZE // 2, AGENT_SIZE // 2), AGENT_SIZE // 2
        )

        self.pos = pygame.math.Vector2(center_x, center_y)
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

        self.rect.center = self.pos

    def collide_walls(self, walls: List[Wall]):
        neligible = 0.1

        for wall in walls:
            if self.rect.colliderect(wall.rect):

                if (
                    self.rect.right > wall.rect.left
                    and self.rect.left < wall.rect.left
                    and self.vel.x > 0
                ):
                    print("collide right")
                    self.pos.x = wall.rect.left - AGENT_SIZE // 2
                if (
                    self.rect.left < wall.rect.right
                    and self.rect.right > wall.rect.right
                    and self.vel.x < 0
                ):
                    print("collide left")
                    self.pos.x = wall.rect.right + AGENT_SIZE // 2
                if (
                    self.rect.bottom > wall.rect.top
                    and self.rect.top < wall.rect.top
                    and self.vel.y > 0
                ):
                    print("collide bottom")
                    self.pos.y = wall.rect.top - AGENT_SIZE // 2
                if (
                    self.rect.top < wall.rect.bottom
                    and self.rect.bottom > wall.rect.bottom
                    and self.vel.y < 0
                ):
                    print("collide top")
                    self.pos.y = wall.rect.bottom + AGENT_SIZE // 2

                self.rect.center = self.pos
                self.vel = pygame.math.Vector2(0, 0)

                return


agent1 = Agent(ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2)

room1 = Room(ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2)
room2 = Room(WIDTH - ROOM_SIZE // 2, ROOM_SIZE // 2)

goal1 = Goal(room2.rect.centerx, room2.rect.centery, GOAL_BLUE)

corridor1 = Corridor(
    room1.rect.center,
    room2.rect.center,
    CorridorOrientation.HORIZONTAL,
    CorridorPosition.TOPLEFT,
)
corridor2 = Corridor(
    room2.rect.center,
    room1.rect.center,
    CorridorOrientation.VERTICAL,
    CorridorPosition.TOPLEFT,
)
corridor3 = Corridor(
    room1.rect.center,
    room2.rect.center,
    CorridorOrientation.HORIZONTAL,
    CorridorPosition.BOTTOMRIGHT,
)
corridor4 = Corridor(
    room2.rect.center,
    room1.rect.center,
    CorridorOrientation.VERTICAL,
    CorridorPosition.BOTTOMRIGHT,
)

WALL_OUTER_LENGTH = HEIGHT - ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2
WALL_INNER_LENGTH = WALL_OUTER_LENGTH - CORRIDOR_WIDTH


walls = pygame.sprite.Group(
    Wall(
        (
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (
            WIDTH - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            WIDTH - ROOM_SIZE + (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            ROOM_SIZE,
            HEIGHT - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (
            ROOM_SIZE,
            HEIGHT - ROOM_SIZE + (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
)


all_sprites = pygame.sprite.Group(
    room1,
    room2,
    goal1,
    corridor1,
    corridor2,
    corridor3,
    corridor4,
    agent1,
)

map_screen = pygame.Surface((WIDTH, HEIGHT))
info_screen = pygame.Surface((WIDTH + 2 * OFFSET, INFO_HEIGHT))

screen.fill(BG_COLOR)
map_screen.fill(BG_COLOR)
info_screen.fill(BG_COLOR)


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.blit(map_screen, (OFFSET, INFO_HEIGHT))
    screen.blit(info_screen, (0, 0))

    agent1.move()
    agent1.collide_walls(walls.sprites())

    all_sprites.draw(map_screen)

    walls.draw(map_screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
