import pygame
import math
import random
import sys
from enum import Enum
from typing import List

# from pygame import gfxdraw

pygame.init()

WIDTH = 600
HEIGHT = 600
OFFSET = 100
INFO_HEIGHT = 220
INFO_MARGIN = 40
WINDOW_WIDTH = WIDTH + 2 * OFFSET
WINDOW_HEIGHT = HEIGHT + OFFSET + INFO_HEIGHT

AGENT_SIZE = 60
ROOM_SIZE = 260
GOAL_SIZE = 50
CORRIDOR_WIDTH = AGENT_SIZE + 40
WALL_WIDTH = 2

AGENT_POS = (ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2)
NPC_POS = (WIDTH - ROOM_SIZE // 2, ROOM_SIZE // 2)

BG_COLOR = (10, 16, 21)
AGENT_BLUE = (70, 180, 180)
AGENT_GREEN = (111, 200, 96)
GOAL_BLUE = (60, 150, 150)
ROOM_COLOR = (45, 50, 56)
CORRIDOR_COLOR = (80, 85, 90)
WALL_COLOR = (60, 67, 75)
INFO_BG_COLOR = (30, 33, 36)
INFO_TEXT_COLOR = (220, 220, 220)
INFO_TEXT_COLOR2 = (145, 150, 155)
LASER_COLOR = (100, 105, 115)
LASER_POINT_COLOR = (200, 60, 60)

FPS = 60

ACC = 1.2
FRIC = -0.12
MAX_SPEED = 6
NPC_VEL = 6

LIDAR_ANGLE = 360
LIDAR_INTERVAL = 5
LIDAR_RANGE = 1200


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

        self.start = start

        if orientation == CorridorOrientation.HORIZONTAL:
            width = length
            height = WALL_WIDTH
            center_x = start[0] + length / 2
            center_y = start[1]
            self.end = (start[0] + length, start[1])
        else:
            width = WALL_WIDTH
            height = length
            center_x = start[0]
            center_y = start[1] + length / 2
            self.end = (start[0], start[1] + length)
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

        # self.acc += self.vel * FRIC
        # self.vel += self.acc

        # if self.vel.length() > MAX_SPEED:
        #     self.vel.scale_to_length(MAX_SPEED)

        # x = v0t + 1/2at^2
        # self.pos += self.vel + 0.5 * self.acc

        self.pos += self.acc * 3

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

    def collide_npc(self, npc):
        if self.rect.colliderect(npc.rect):
            self.vel = pygame.math.Vector2(0, 0)
            return True
        return False

    def check_goal(self, goal: Goal):
        if self.rect.colliderect(goal.rect):
            return True
        return False

    def create_lasers(self):
        lasers = []
        for angle in range(0, -LIDAR_ANGLE, -LIDAR_INTERVAL):
            laser = (
                (self.pos.x, self.pos.y),
                (
                    self.pos.x + LIDAR_RANGE * math.cos(math.radians(angle)),
                    self.pos.y + LIDAR_RANGE * math.sin(math.radians(angle)),
                ),
            )
            lasers.append(laser)
        return lasers

    def distance_from_goal(self, goal):
        return math.sqrt(
            (self.pos.x - goal.rect.center[0]) ** 2
            + (self.pos.y - goal.rect.center[1]) ** 2
        )


class NPC(pygame.sprite.Sprite):
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
                self.pos.x -= NPC_VEL
            elif self.pos.y < HEIGHT - ROOM_SIZE // 2:
                self.pos.y += NPC_VEL
        else:
            if self.pos.y < HEIGHT - ROOM_SIZE // 2:
                self.pos.y += NPC_VEL
            elif self.pos.x > ROOM_SIZE // 2:
                self.pos.x -= NPC_VEL

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
npc = NPC(AGENT_GREEN)


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

players = pygame.sprite.Group(agent, npc)

screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
map_screen = pygame.Surface((WIDTH, HEIGHT))
info_screen = pygame.Surface((WINDOW_WIDTH, INFO_HEIGHT))


def line_intersect(p0, p1, q0, q1):
    """
    2つの直線PとQの交点を計算
    pからp+rに向かう直線とqからq+sに向かう直線が
    p+trおよびq+usで交わるようなtとuを求める
    """
    det = (p1[0] - p0[0]) * (q1[1] - q0[1]) + (p1[1] - p0[1]) * (q0[0] - q1[0])
    if det != 0:
        t = (
            (q0[0] - p0[0]) * (q1[1] - q0[1]) + (q0[1] - p0[1]) * (q0[0] - q1[0])
        ) / det
        u = (
            (q0[0] - p0[0]) * (p1[1] - p0[1]) + (q0[1] - p0[1]) * (p0[0] - p1[0])
        ) / det
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
    return None


def laser_scan(laser, obstacle_lines):
    min_intersection = laser[1]
    min_distance = LIDAR_RANGE
    for obstacle in obstacle_lines:
        intersection = line_intersect(laser[0], laser[1], obstacle[0], obstacle[1])
        if intersection is not None:
            distance = math.sqrt(
                (intersection[0] - laser[0][0]) ** 2
                + (intersection[1] - laser[0][1]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                min_intersection = intersection

    return min_intersection


def get_obstacle_lines(walls, npc):
    lines = []
    for wall in walls:
        lines.append((wall.start, wall.end))
    lines.append((npc.rect.topleft, npc.rect.topright))
    lines.append((npc.rect.topleft, npc.rect.bottomleft))
    lines.append((npc.rect.topright, npc.rect.bottomright))
    lines.append((npc.rect.bottomleft, npc.rect.bottomright))
    return lines


def reset():
    pygame.display.flip()
    pygame.time.wait(200)
    agent.reset(AGENT_POS)
    npc.reset(NPC_POS)


if __name__ == "__main__":
    reset()

    episode = 0
    episode_font = pygame.font.SysFont("ubuntu", 30)
    result_font = pygame.font.SysFont("ubuntu", 45)
    info_font = pygame.font.SysFont("ubuntu", 22)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill(BG_COLOR)
        map_screen.fill(BG_COLOR)
        info_screen.fill(BG_COLOR)

        maps.draw(map_screen)
        walls.draw(map_screen)

        agent.move()
        npc.auto_move()

        lasers = agent.create_lasers()
        obstacle_lines = get_obstacle_lines(walls, npc)
        for laser in lasers:
            intersection = laser_scan(laser, obstacle_lines)
            pygame.draw.line(map_screen, LASER_COLOR, agent.pos, intersection)
            pygame.draw.circle(map_screen, LASER_POINT_COLOR, intersection, 2.5)

        players.draw(map_screen)

        episode_text = episode_font.render(f"Episode: {episode}", True, INFO_TEXT_COLOR)
        timestep_text = distance_text = info_font.render(
            f"Timestep: 0", True, INFO_TEXT_COLOR2
        )
        distance_text = info_font.render(
            f"Distance from goal: {agent.distance_from_goal(goal1):.0f}",
            True,
            INFO_TEXT_COLOR2,
        )
        pygame.draw.rect(
            info_screen,
            INFO_BG_COLOR,
            (
                INFO_MARGIN,
                INFO_MARGIN,
                WINDOW_WIDTH - 2 * INFO_MARGIN,
                INFO_HEIGHT - 2 * INFO_MARGIN,
            ),
            0,
            20,
        )
        info_screen.blit(episode_text, (INFO_MARGIN + 24, INFO_MARGIN + 20))
        info_screen.blit(timestep_text, (INFO_MARGIN + 24, INFO_MARGIN + 65))
        info_screen.blit(distance_text, (INFO_MARGIN + 24, INFO_MARGIN + 95))

        screen.blit(map_screen, (OFFSET, INFO_HEIGHT))
        screen.blit(info_screen, (0, 0))

        pygame.display.flip()

        if agent.check_goal(goal1):
            result_text = result_font.render("CLEAR !!", True, INFO_TEXT_COLOR)
            screen.blit(result_text, (400, 80))
            reset()
            episode += 1

        if agent.collide_npc(npc) or agent.collide_walls(walls.sprites()):
            result_text = result_font.render("FAILED", True, INFO_TEXT_COLOR)
            screen.blit(result_text, (400, 80))
            reset()
            episode += 1

    pygame.quit()
    sys.exit()
