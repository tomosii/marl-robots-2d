from enum import Enum
import math
import random
import numpy as np
import pygame
from typing import List, Tuple

from envs.crossroads.utils import line_intersect


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
    ):
        super(Agent, self).__init__()
        super(pygame.sprite.Sprite, self).__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, color, (size // 2, size // 2), size // 2)

        self.movable = True
        self.sendable = False

        self.VEL = velocity

        self.pos = pygame.math.Vector2(0, 0)
        self.vel = pygame.math.Vector2(0, 0)
        # self.acc = pygame.math.Vector2(0, 0)

    def move(self, direction: Direction):
        self.vel = pygame.math.Vector2(0, 0)

        if direction == Direction.LEFT:
            self.vel.x = -self.VEL
        if direction == Direction.RIGHT:
            self.vel.x = self.VEL

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
