import pygame
import random
from enum import Enum


class CorridorOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class CorridorPosition(Enum):
    TOPLEFT = 0
    BOTTOMRIGHT = 1


class Room(pygame.sprite.Sprite):
    def __init__(self, center_x, center_y, size, color):
        super().__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)

        pygame.draw.rect(self.image, color, [0, 0, size, size])

        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Goal(pygame.sprite.Sprite):
    def __init__(self, center_x, center_y, size, color):
        super().__init__()
        self.image = pygame.Surface([size, size], pygame.SRCALPHA)

        pygame.draw.rect(self.image, color, [0, 0, size, size], 0, 5)

        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Wall(pygame.sprite.Sprite):
    def __init__(self, start, length, orientation, wall_width, color):
        super().__init__()

        self.start = start

        if orientation == CorridorOrientation.HORIZONTAL:
            width = length
            height = wall_width
            center_x = start[0] + length / 2
            center_y = start[1]
            self.end = (start[0] + length, start[1])
        else:
            width = wall_width
            height = length
            center_x = start[0]
            center_y = start[1] + length / 2
            self.end = (start[0], start[1] + length)
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)
        pygame.draw.rect(self.image, color, [0, 0, width, height])
        self.rect = self.image.get_rect()
        self.rect.center = (center_x, center_y)


class Corridor(pygame.sprite.Sprite):
    def __init__(
        self,
        start_room,
        end_room,
        orientation,
        position,
        corridor_width,
        room_size,
        color,
    ):
        super().__init__()
        if orientation == CorridorOrientation.HORIZONTAL:
            width = end_room[0] - start_room[0] - room_size // 2 + corridor_width // 2
            height = corridor_width
            if position == CorridorPosition.TOPLEFT:
                left = start_room[0] - corridor_width // 2
                top = end_room[1] - corridor_width // 2
            else:
                left = start_room[0] + room_size // 2
                top = start_room[1] - corridor_width // 2
        else:
            width = corridor_width
            height = end_room[1] - start_room[1] - room_size // 2 + corridor_width // 2
            if position == CorridorPosition.TOPLEFT:
                left = end_room[0] - corridor_width // 2
                top = start_room[1] - corridor_width // 2
            else:
                left = start_room[0] - corridor_width // 2
                top = start_room[1] + room_size // 2

        self.image = pygame.Surface([width, height], pygame.SRCALPHA)

        self.rect = pygame.draw.rect(self.image, color, [0, 0, width, height], 0, 0)
        self.rect.left = left
        self.rect.top = top


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

    def reset(self, pos, random_direction=None):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.rect.center = self.pos

        if random_direction:
            rand = random.randint(0, 1)
            if rand == 0:
                self.path = CorridorOrientation.HORIZONTAL
            else:
                self.path = CorridorOrientation.VERTICAL
        else:
            self.path = CorridorOrientation.HORIZONTAL
