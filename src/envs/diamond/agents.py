from enum import Enum
import math
import random
import numpy as np
import pygame
from typing import List, Tuple

from envs.diamond.utils import line_intersect


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Agent:
    def __init__(self):
        # 動けるか
        self.movable: bool = None
        # メッセージを送信できるか
        self.sendable: bool = None

    def reset(self):
        raise NotImplementedError

    def move(self, direction: Direction):
        raise NotImplementedError


class RobotAgent(Agent, pygame.sprite.Sprite):
    """
    ロボットエージェント (RA)
    周りをLiDARで観測しながら移動する
    """

    def __init__(
        self,
        color: Tuple[int, int, int],
        size: int,
        velocity: float,
        lidar_range: float,
        lidar_angle: float,
        lidar_interval: float,
    ):
        super(Agent, self).__init__()
        super(pygame.sprite.Sprite, self).__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        # pygame.draw.circle(self.image, color, (size // 2, size // 2), size // 2)
        pygame.draw.rect(self.image, color, [0, 0, size, size], 0, 3)

        self.movable = True
        self.sendable = False

        self.VEL = velocity
        self.LIDAR_RANGE = lidar_range
        self.LIDAR_ANGLE = lidar_angle
        self.LIDAR_INTERVAL = lidar_interval

        self.pos = pygame.math.Vector2(0, 0)
        self.vel = pygame.math.Vector2(0, 0)
        # self.acc = pygame.math.Vector2(0, 0)

    def move(self, direction: Direction):
        self.vel = pygame.math.Vector2(0, 0)

        if direction == Direction.LEFT:
            self.vel.x = -self.VEL
        if direction == Direction.RIGHT:
            self.vel.x = self.VEL
        if direction == Direction.UP:
            self.vel.y = -self.VEL
        if direction == Direction.DOWN:
            self.vel.y = self.VEL

        # self.acc += self.vel * self.FRIC
        # self.vel += self.acc

        # if self.vel.length() > self.MAX_SPEED:
        #     self.vel.scale_to_length(self.MAX_SPEED)

        # x = v0t + 1/2at^2
        # self.pos += self.vel + 0.5 * self.acc

        self.pos += self.vel

        self.rect.center = self.pos

    def reset(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.rect.center = self.pos

    def __create_lasers(self) -> np.ndarray:
        lasers = []
        for angle in range(0, -(self.LIDAR_ANGLE - 1), -self.LIDAR_INTERVAL):
            laser = (
                (self.pos.x, self.pos.y),
                (
                    self.pos.x + self.LIDAR_RANGE * math.cos(math.radians(angle)),
                    self.pos.y + self.LIDAR_RANGE * math.sin(math.radians(angle)),
                ),
            )
            lasers.append(laser)
        return np.array(lasers)

    def laser_scan(self, obstacle_lines: List) -> np.ndarray:
        """
        ライダーのレーザーを飛ばした際の障害物との交点を求める
        """
        intersections = []
        lasers = self.__create_lasers()

        for laser in lasers:
            min_intersection = laser[1]
            min_distance = self.LIDAR_RANGE
            for obstacle in obstacle_lines:
                intersection = line_intersect(
                    laser[0], laser[1], obstacle[0], obstacle[1]
                )
                if intersection is not None:
                    distance = math.sqrt(
                        (intersection[0] - laser[0][0]) ** 2
                        + (intersection[1] - laser[0][1]) ** 2
                    )
                    if distance < min_distance:
                        min_distance = distance
                        min_intersection = intersection

            intersections.append(min_intersection)
        return np.array(intersections)


class SensorAgent(Agent):
    """
    センサーエージェント (SA)
    動くことはできないが、全体を観測してメッセージを送信することができる
    """

    def __init__(self):
        self.movable = False
        self.sendable = True

    def reset(self):
        return

    def move(self, direction: Direction):
        print("SensorAgent can't move")
        return
