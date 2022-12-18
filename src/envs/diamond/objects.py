from typing import Tuple
import pygame
import random
from enum import Enum


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Room(pygame.sprite.Sprite):
    def __init__(
        self,
        width,
        height,
        color,
    ):
        super().__init__()
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)

        pygame.draw.rect(self.image, color, [0, 0, width, height])

        self.rect = self.image.get_rect()


class Goal(pygame.sprite.Sprite):
    def __init__(
        self,
        center_x,
        center_y,
        size,
        color,
    ):
        super().__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)

        pygame.draw.rect(self.image, color, [0, 0, size, size], 0, 5)

        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Wall(pygame.sprite.Sprite):
    def __init__(
        self,
        x,
        y,
        width,
        height,
        color,
    ):
        super().__init__()
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)

        pygame.draw.rect(self.image, color, [0, 0, width, height], 0, 0)

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Border(pygame.sprite.Sprite):
    def __init__(
        self,
        start,
        length,
        orientation,
        border_width,
        color,
    ):
        super().__init__()

        self.start = start

        if orientation == Orientation.HORIZONTAL:
            width = length
            height = border_width
            center_x = start[0] + length / 2
            center_y = start[1]
            self.end = (start[0] + length, start[1])
        else:
            width = border_width
            height = length
            center_x = start[0]
            center_y = start[1] + length / 2
            self.end = (start[0], start[1] + length)
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)
        pygame.draw.rect(self.image, color, [0, 0, width, height])
        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Guard(pygame.sprite.Sprite):
    def __init__(
        self,
        color: Tuple[int, int, int],
        size: int,
        velocity: float,
    ):
        super().__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, color, (size // 2, size // 2), size // 2)

        self.VEL = velocity

        self.pos = pygame.math.Vector2(0, 0)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)

    def patrol(self, corner1, goal1, corner2, goal2):
        if self.path == Orientation.HORIZONTAL:
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

    def reset(self, pos, random_direction=None):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.rect.center = self.pos

        if random_direction:
            rand = random.randint(0, 1)
            if rand == 0:
                self.path = Orientation.HORIZONTAL
            else:
                self.path = Orientation.VERTICAL
        else:
            self.path = Orientation.HORIZONTAL
