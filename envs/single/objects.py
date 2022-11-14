import pygame
from enum import Enum


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
