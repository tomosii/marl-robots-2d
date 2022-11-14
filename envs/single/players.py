import pygame
import math
import random
from typing import List


class Agent(pygame.sprite.Sprite):
    def __init__(
        self,
        color,
        size,
    ):
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

        self.acc += self.vel * FRIC
        self.vel += self.acc

        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)

        # x = v0t + 1/2at^2
        self.pos += self.vel + 0.5 * self.acc

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
