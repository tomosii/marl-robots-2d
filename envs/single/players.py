import pygame
import math
import random
from typing import List, Tuple

from objects import CorridorOrientation, Goal, Wall
from utils import line_intersect


class Agent(pygame.sprite.Sprite):
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
        super().__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, color, (size // 2, size // 2), size // 2)

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

        pressed_keys = pygame.key.get_pressed()

        if action == 0:
            self.acc.x = -self.ACC
        if action == 1:
            self.acc.x = self.ACC
        if action == 2:
            self.acc.y = -self.ACC
        if action == 3:
            self.acc.y = self.ACC

        self.acc += self.vel * self.FRIC
        self.vel += self.acc

        if self.vel.length() > self.MAX_SPEED:
            self.vel.scale_to_length(self.MAX_SPEED)

        # x = v0t + 1/2at^2
        self.pos += self.vel + 0.5 * self.acc

        self.rect.center = self.pos

    def reset(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.rect.center = self.pos

    def collide_npc(self, npc):
        if self.rect.colliderect(npc.rect):
            self.vel = pygame.math.Vector2(0, 0)
            return True
        return False

    def check_goal(self, goal: Goal):
        if self.rect.colliderect(goal.rect):
            return True
        return False

    def create_lasers(self) -> List:
        lasers = []
        for angle in range(0, -self.LIDAR_ANGLE, -self.LIDAR_INTERVAL):
            laser = (
                (self.pos.x, self.pos.y),
                (
                    self.pos.x + self.LIDAR_RANGE * math.cos(math.radians(angle)),
                    self.pos.y + self.LIDAR_RANGE * math.sin(math.radians(angle)),
                ),
            )
            lasers.append(laser)
        return lasers

    def laser_scan(self, laser, obstacle_lines) -> Tuple[float, float]:
        """
        レーザーが障害物に当たる位置を計算
        """

        min_intersection = laser[1]
        min_distance = self.LIDAR_RANGE
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

    def distance_from_goal(self, goal):
        return math.sqrt(
            (self.pos.x - goal.rect.center[0]) ** 2
            + (self.pos.y - goal.rect.center[1]) ** 2
        )


class NPC(pygame.sprite.Sprite):
    def __init__(self, color, size, velocity):
        super().__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, color, (size // 2, size // 2), size // 2)

        self.VEL = velocity

        self.pos = pygame.math.Vector2(0, 0)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)

    def auto_move(self, corner1, goal1, corner2, goal2):
        if self.path == CorridorOrientation.HORIZONTAL:
            if self.pos.x > corner1:
                self.pos.x -= self.VEL
            elif self.pos.y < goal1:
                self.pos.y += self.VEL
        else:
            if self.pos.y < corner2:
                self.pos.y += self.VEL
            elif self.pos.x > goal2:
                self.pos.x -= self.VEL

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
