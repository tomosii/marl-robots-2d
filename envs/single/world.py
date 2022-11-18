import math
import numpy as np
import pygame
from typing import List, Tuple

from envs.single.objects import (
    Corridor,
    CorridorOrientation,
    CorridorPosition,
    Goal,
    Room,
    Wall,
)
from envs.single.players import NPC, Agent


class World:
    def __init__(self):
        self.agent: Agent = None
        self.npc: NPC = None
        self.goal: Goal = None

        self.players: pygame.sprite.Group = None
        self.maps: pygame.sprite.Group = None
        self.walls: List[Wall] = None

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_obstacle_lines(self) -> List:
        """
        レーザーとの交点を計算するために、すべての障害物の線分を取得する
        """
        lines = []
        for wall in self.walls:
            lines.append((wall.start, wall.end))
        lines.append((self.npc.rect.topleft, self.npc.rect.topright))
        lines.append((self.npc.rect.topleft, self.npc.rect.bottomleft))
        lines.append((self.npc.rect.topright, self.npc.rect.bottomright))
        lines.append((self.npc.rect.bottomleft, self.npc.rect.bottomright))
        return lines

    def draw(self, screen: pygame.Surface):
        raise NotImplementedError

    def check_collision(self) -> bool:
        raise NotImplementedError

    def check_goal(self) -> bool:
        raise NotImplementedError

    def get_distance_from_goal(self):
        return math.sqrt(
            (self.agent.pos.x - self.goal.rect.center[0]) ** 2
            + (self.agent.pos.y - self.goal.rect.center[1]) ** 2
        )


class SimpleWorld(World):
    WIDTH = 600
    HEIGHT = 600

    AGENT_SIZE = 60
    ROOM_SIZE = 260
    GOAL_SIZE = 50
    CORRIDOR_WIDTH = AGENT_SIZE + 40
    WALL_WIDTH = 2
    WALL_OUTER_LENGTH = HEIGHT - ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2
    WALL_INNER_LENGTH = WALL_OUTER_LENGTH - CORRIDOR_WIDTH
    ROOM_WALL_LENGTH = (ROOM_SIZE - CORRIDOR_WIDTH) // 2

    ROOM_COLOR = (45, 50, 56)
    CORRIDOR_COLOR = (80, 85, 90)
    WALL_COLOR = (60, 67, 75)
    GOAL_BLUE = (60, 150, 150)
    AGENT_BLUE = (70, 180, 180)
    AGENT_GREEN = (111, 200, 96)
    LASER_COLOR = (100, 105, 115)
    LASER_POINT_COLOR = (200, 60, 60)

    AGENT_POS = (ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2)
    NPC_POS = (WIDTH - ROOM_SIZE // 2, ROOM_SIZE // 2)

    ACC = 3
    FRIC = -0.12
    MAX_SPEED = 6
    NPC_VEL = 3

    LIDAR_ANGLE = 360
    LIDAR_INTERVAL = 30
    LIDAR_RANGE = math.sqrt(WIDTH**2 + HEIGHT**2)

    room1 = Room(ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2, ROOM_SIZE, ROOM_COLOR)
    room2 = Room(WIDTH - ROOM_SIZE // 2, ROOM_SIZE // 2, ROOM_SIZE, ROOM_COLOR)
    # goal = Goal(room2.rect.centerx, room2.rect.centery, GOAL_SIZE, GOAL_BLUE)
    goal = Goal(room2.rect.centerx, room2.rect.centery, GOAL_SIZE, GOAL_BLUE)
    maps = pygame.sprite.Group(
        room1,
        room2,
        Corridor(
            room1.rect.center,
            room2.rect.center,
            CorridorOrientation.HORIZONTAL,
            CorridorPosition.TOPLEFT,
            CORRIDOR_WIDTH,
            ROOM_SIZE,
            CORRIDOR_COLOR,
        ),
        Corridor(
            room2.rect.center,
            room1.rect.center,
            CorridorOrientation.VERTICAL,
            CorridorPosition.TOPLEFT,
            CORRIDOR_WIDTH,
            ROOM_SIZE,
            CORRIDOR_COLOR,
        ),
        Corridor(
            room1.rect.center,
            room2.rect.center,
            CorridorOrientation.HORIZONTAL,
            CorridorPosition.BOTTOMRIGHT,
            CORRIDOR_WIDTH,
            ROOM_SIZE,
            CORRIDOR_COLOR,
        ),
        Corridor(
            room2.rect.center,
            room1.rect.center,
            CorridorOrientation.VERTICAL,
            CorridorPosition.BOTTOMRIGHT,
            CORRIDOR_WIDTH,
            ROOM_SIZE,
            CORRIDOR_COLOR,
        ),
        goal,
    )

    walls = pygame.sprite.Group(
        Wall(
            (
                (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
                (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ),
            WALL_OUTER_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (
                (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
                (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ),
            WALL_OUTER_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (
                ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
                ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ),
            WALL_INNER_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (
                ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
                ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ),
            WALL_INNER_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (
                WIDTH - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
                ROOM_SIZE,
            ),
            WALL_OUTER_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (
                WIDTH - ROOM_SIZE + (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
                ROOM_SIZE,
            ),
            WALL_INNER_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (
                ROOM_SIZE,
                HEIGHT - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ),
            WALL_OUTER_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (
                ROOM_SIZE,
                HEIGHT - ROOM_SIZE + (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ),
            WALL_INNER_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (0, HEIGHT - ROOM_SIZE),
            ROOM_SIZE,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (0, HEIGHT),
            ROOM_SIZE,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (WIDTH, 0),
            ROOM_SIZE,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (WIDTH - ROOM_SIZE, 0),
            ROOM_SIZE,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (0, HEIGHT - ROOM_SIZE),
            ROOM_WALL_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (ROOM_SIZE - ROOM_WALL_LENGTH, HEIGHT - ROOM_SIZE),
            ROOM_WALL_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (ROOM_SIZE, HEIGHT - ROOM_SIZE),
            ROOM_WALL_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (ROOM_SIZE, HEIGHT - ROOM_WALL_LENGTH),
            ROOM_WALL_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (WIDTH - ROOM_SIZE, 0),
            ROOM_WALL_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (WIDTH - ROOM_SIZE, ROOM_SIZE - ROOM_WALL_LENGTH),
            ROOM_WALL_LENGTH,
            CorridorOrientation.VERTICAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (WIDTH - ROOM_SIZE, ROOM_SIZE),
            ROOM_WALL_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
        Wall(
            (WIDTH - ROOM_WALL_LENGTH, ROOM_SIZE),
            ROOM_WALL_LENGTH,
            CorridorOrientation.HORIZONTAL,
            WALL_WIDTH,
            WALL_COLOR,
        ),
    )

    def __init__(self):
        self.agent = Agent(
            self.AGENT_BLUE,
            self.AGENT_SIZE,
            self.ACC,
            self.FRIC,
            self.MAX_SPEED,
            self.LIDAR_RANGE,
            self.LIDAR_ANGLE,
            self.LIDAR_INTERVAL,
        )
        self.npc = NPC(self.AGENT_GREEN, self.AGENT_SIZE, self.NPC_VEL)
        self.players = pygame.sprite.Group(self.agent, self.npc)

        self.scan_points = []

    def reset(self, random_direction=False):
        self.agent.reset(self.AGENT_POS)
        self.npc.reset(self.NPC_POS, random_direction=random_direction)

    def step(self, action):
        self.agent.move(action)
        self.npc.auto_move(
            self.ROOM_SIZE // 2,
            self.HEIGHT - self.ROOM_SIZE // 2,
            self.HEIGHT - self.ROOM_SIZE // 2,
            self.ROOM_SIZE // 2,
        )

    def draw(self, screen):
        self.maps.draw(screen)
        self.walls.draw(screen)
        self.__draw_lasers(screen)
        self.players.draw(screen)

    def __draw_lasers(self, screen):
        for intersection in self.laser_points:
            pygame.draw.line(screen, self.LASER_COLOR, self.agent.pos, intersection)
            pygame.draw.circle(screen, self.LASER_POINT_COLOR, intersection, 2.5)

    def check_collision(self):
        return pygame.sprite.spritecollideany(
            self.agent, self.walls
        ) or pygame.sprite.collide_rect(self.agent, self.npc)

    def check_goal(self):
        return pygame.sprite.collide_rect(self.agent, self.goal)

    def get_num_lasers(self) -> int:
        return self.LIDAR_ANGLE // self.LIDAR_INTERVAL

    def laser_scan(self) -> np.ndarray:
        self.laser_points = self.agent.laser_scan(self.get_obstacle_lines())
        laser_distances = [
            math.sqrt(
                (point[0] - self.agent.pos.x) ** 2 + (point[1] - self.agent.pos.y) ** 2
            )
            for point in self.laser_points
        ]
        return np.array(laser_distances)

    def normalize_distances(self, distances: np.ndarray) -> np.ndarray:
        return distances / self.LIDAR_RANGE

    def get_relative_normalized_goal_position(self) -> np.ndarray:
        return np.array(
            [
                (self.goal.rect.centerx - self.agent.pos.x) / self.WIDTH,
                (self.goal.rect.centery - self.agent.pos.y) / self.HEIGHT,
            ]
        )

    def get_normalized_distance_from_goal(self) -> float:
        return self.get_distance_from_goal() / self.LIDAR_RANGE
