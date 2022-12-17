import pygame
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
