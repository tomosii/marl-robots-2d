import math
import random
import numpy as np
import pygame
from typing import List, Tuple

from envs.multi.objects import CorridorOrientation, Goal, Wall
from envs.multi.utils import line_intersect


class Agent:
    def __init__(self):
        self.movable: bool = None
        self.sendable: bool = None

    def reset(self):
        pass


class RobotAgent(Agent, pygame.sprite.Sprite):
    def __init__(
        self,
        color,
        size,
        acceleration,
        friction,
        max_speed,
        lidar_range,
        lidar_angle,
        lidar_interval,
    ):
        super(Agent, self).__init__()
        super(pygame.sprite.Sprite, self).__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        # pygame.draw.circle(self.image, color, (size // 2, size // 2), size // 2)
        pygame.draw.rect(self.image, color, [0, 0, size, size], 0, 3)

        self.movable = True
        self.sendable = False

        self.ACC = acceleration
        self.FRIC = friction
        self.MAX_SPEED = max_speed
        self.LIDAR_RANGE = lidar_range
        self.LIDAR_ANGLE = lidar_angle
        self.LIDAR_INTERVAL = lidar_interval

        self.pos = pygame.math.Vector2(0, 0)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)

    def move(self, action):
        self.acc = pygame.math.Vector2(0, 0)

        if action == 0:
            self.acc.x = -self.ACC
        if action == 1:
            self.acc.x = self.ACC
        if action == 2:
            self.acc.y = -self.ACC
        if action == 3:
            self.acc.y = self.ACC

        # self.acc += self.vel * self.FRIC
        # self.vel += self.acc

        # if self.vel.length() > self.MAX_SPEED:
        #     self.vel.scale_to_length(self.MAX_SPEED)

        # x = v0t + 1/2at^2
        # self.pos += self.vel + 0.5 * self.acc

        self.pos += self.acc

        self.rect.center = self.pos

    def reset(self, pos):
        super().reset()
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
    def __init__(self):
        self.message = 0

        self.movable = False
        self.sendable = True

    def reset(self):
        return super().reset()
