import pygame
import math
import random
import sys
from enum import Enum
from typing import List

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

AGENT_POS = (ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2)
OBSTACLE_POS = (WIDTH - ROOM_SIZE // 2, ROOM_SIZE // 2)

BG_COLOR = (12, 14, 14)
AGENT_BLUE = (70, 180, 180)
AGENT_GREEN = (111, 200, 96)
GOAL_BLUE = (50, 140, 140)
ROOM_COLOR = (30, 32, 33)
CORRIDOR_COLOR = (50, 53, 54)
WALL_COLOR = (60, 60, 60)
INFO_BG_COLOR = (30, 30, 30)
INFO_TEXT_COLOR = (220, 220, 220)
LASER_COLOR = (75, 75, 75)

FPS = 60

ACC = 0.7
FRIC = -0.12
MAX_SPEED = 6
OBSTACLE_VEL = 5

LIDAR_ANGLE = 360
LIDAR_INTERVAL = 45

screen = pygame.display.set_mode([WIDTH + 2 * OFFSET, HEIGHT + OFFSET + INFO_HEIGHT])
pygame.display.set_caption("Two Corridors")
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

        pygame.draw.rect(self.image, ROOM_COLOR, [0, 0, ROOM_SIZE, ROOM_SIZE])

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
    def __init__(self, color):
        super().__init__()
        self.image = pygame.Surface([AGENT_SIZE, AGENT_SIZE], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(
            self.image, color, (AGENT_SIZE // 2, AGENT_SIZE // 2), AGENT_SIZE // 2
        )

        self.pos = pygame.math.Vector2(0, 0)
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

    def reset(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.rect.center = self.pos

    def collide_walls(self, walls: List[Wall]):
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                return True
        return False

    def collide_obstacle(self, obstacle):
        if self.rect.colliderect(obstacle.rect):
            self.vel = pygame.math.Vector2(0, 0)
            return True
        return False

    def check_goal(self, goal: Goal):
        if self.rect.colliderect(goal.rect):
            return True
        return False

    def create_lasers(self):
        return [
            (
                self.pos.x + 1200 * math.cos(math.radians(angle)),
                self.pos.y + 1200 * math.sin(math.radians(angle)),
            )
            for angle in range(0, -LIDAR_ANGLE, -LIDAR_INTERVAL)
        ]


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, color):
        super().__init__()
        self.image = pygame.Surface([AGENT_SIZE, AGENT_SIZE], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(
            self.image, color, (AGENT_SIZE // 2, AGENT_SIZE // 2), AGENT_SIZE // 2
        )

        self.pos = pygame.math.Vector2(0, 0)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)

    def auto_move(self):
        if self.path == CorridorOrientation.HORIZONTAL:
            if self.pos.x > ROOM_SIZE // 2:
                self.pos.x -= OBSTACLE_VEL
            elif self.pos.y < HEIGHT - ROOM_SIZE // 2:
                self.pos.y += OBSTACLE_VEL
        else:
            if self.pos.y < HEIGHT - ROOM_SIZE // 2:
                self.pos.y += OBSTACLE_VEL
            elif self.pos.x > ROOM_SIZE // 2:
                self.pos.x -= OBSTACLE_VEL

        self.rect.center = self.pos

    def reset(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.rect.center = self.pos

        rand = random.randint(0, 1)
        if rand == 0:
            self.path = CorridorOrientation.HORIZONTAL
        else:
            self.path = CorridorOrientation.VERTICAL


agent = Agent(AGENT_BLUE)
obstacle = Obstacle(AGENT_GREEN)


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
ROOM_WALL_LENGTH = (ROOM_SIZE - CORRIDOR_WIDTH) // 2


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
    Wall((0, HEIGHT - ROOM_SIZE), ROOM_SIZE, CorridorOrientation.VERTICAL),
    Wall((0, HEIGHT), ROOM_SIZE, CorridorOrientation.HORIZONTAL),
    Wall((WIDTH, 0), ROOM_SIZE, CorridorOrientation.VERTICAL),
    Wall((WIDTH - ROOM_SIZE, 0), ROOM_SIZE, CorridorOrientation.HORIZONTAL),
    Wall((0, HEIGHT - ROOM_SIZE), ROOM_WALL_LENGTH, CorridorOrientation.HORIZONTAL),
    Wall(
        (ROOM_SIZE - ROOM_WALL_LENGTH, HEIGHT - ROOM_SIZE),
        ROOM_WALL_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (ROOM_SIZE, HEIGHT - ROOM_SIZE), ROOM_WALL_LENGTH, CorridorOrientation.VERTICAL
    ),
    Wall(
        (ROOM_SIZE, HEIGHT - ROOM_WALL_LENGTH),
        ROOM_WALL_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall((WIDTH - ROOM_SIZE, 0), ROOM_WALL_LENGTH, CorridorOrientation.VERTICAL),
    Wall(
        (WIDTH - ROOM_SIZE, ROOM_SIZE - ROOM_WALL_LENGTH),
        ROOM_WALL_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (WIDTH - ROOM_SIZE, ROOM_SIZE), ROOM_WALL_LENGTH, CorridorOrientation.HORIZONTAL
    ),
    Wall(
        (WIDTH - ROOM_WALL_LENGTH, ROOM_SIZE),
        ROOM_WALL_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
)


maps = pygame.sprite.Group(
    room1,
    room2,
    goal1,
    corridor1,
    corridor2,
    corridor3,
    corridor4,
)

agents = pygame.sprite.Group(agent, obstacle)

map_screen = pygame.Surface((WIDTH, HEIGHT))
info_screen = pygame.Surface((WIDTH + 2 * OFFSET, INFO_HEIGHT))


def line_intersect(p0, p1, q0, q1):

    pass


def reset():
    pygame.display.flip()
    pygame.time.wait(200)
    agent.reset(AGENT_POS)
    obstacle.reset(OBSTACLE_POS)


if __name__ == "__main__":
    reset()

    episode = 0
    font = pygame.font.SysFont("Arial", 30)
    result_font = pygame.font.SysFont("Arial", 45)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        episode_text = font.render(f"Episode: {episode}", True, INFO_TEXT_COLOR)

        screen.fill(BG_COLOR)
        map_screen.fill(BG_COLOR)
        info_screen.fill(BG_COLOR)

        maps.draw(map_screen)
        walls.draw(map_screen)

        agent.move()
        obstacle.auto_move()

        lasers = agent.create_lasers()
        for laser in lasers:
            intersection = lidar_wall_intersect(P)
            pygame.draw.line(map_screen, LASER_COLOR, agent.pos, intersection)

        agents.draw(map_screen)

        screen.blit(map_screen, (OFFSET, INFO_HEIGHT))
        screen.blit(info_screen, (0, 0))
        screen.blit(episode_text, (40, 80))

        pygame.display.flip()

        if agent.check_goal(goal1):
            result_text = result_font.render("CLEAR !!", True, INFO_TEXT_COLOR)
            screen.blit(result_text, (400, 80))
            reset()
            episode += 1

        if agent.collide_obstacle(obstacle) or agent.collide_walls(walls.sprites()):
            result_text = result_font.render("FAILED", True, INFO_TEXT_COLOR)
            screen.blit(result_text, (400, 80))
            reset()
            episode += 1

    pygame.quit()
    sys.exit()
