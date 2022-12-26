import math
import numpy as np
import pygame
import random
from typing import List, Tuple

from envs.crossroads.objects import Goal, Room, Wall, Border, Guard, Orientation
from envs.crossroads.agents import RobotAgent, SensorAgent, Agent
from envs.crossroads.utils import one_hot_encode


class World:
    def __init__(self):
        self.channel = None
        self.players: pygame.sprite.Group = None
        self.maps: pygame.sprite.Group = None
        self.walls: List[Wall] = None

    def reset(self):
        raise NotImplementedError

    def draw(self, screen: pygame.Surface):
        raise NotImplementedError

    def check_collision(self) -> bool:
        raise NotImplementedError

    def check_goal(self) -> bool:
        raise NotImplementedError


class CrossroadsWorld(World):
    AGENT_SIZE = 30
    GOAL_SIZE = 50

    ROAD_WIDTH = 500
    ROAD_HEIGHT = 150
    GOAL_OFFSET = 40

    BORDER_WIDTH = 2

    WALL_COLOR = (45, 50, 56)
    FLOOR_COLOR = (80, 85, 90)
    BORDER_COLOR = (60, 67, 75)
    GOAL_TRUE_COLOR = (150, 65, 72)
    GOAL_FALSE_COLOR = (60, 65, 72)

    AGENT_COLOR = (70, 180, 180)

    def __init__(
        self,
        agent_velocity: float,
        channel_size: int,
    ):

        self.agent_velocity = agent_velocity
        self.channel_size = channel_size

        self.WIDTH = self.ROAD_WIDTH
        self.HEIGHT = self.ROAD_HEIGHT

        self.r_agent = RobotAgent(
            self.AGENT_COLOR,
            self.AGENT_SIZE,
            self.agent_velocity,
        )
        self.s_agent = SensorAgent()
        self.agents: List[Agent] = [self.r_agent, self.s_agent]

        self.AGENT_POS = (
            self.WIDTH // 2,
            self.HEIGHT // 2,
        )

        self.channel = None

        self.true_goal = 0
        self.prev_true_goal = 0

        self.players = pygame.sprite.Group(self.r_agent)

        self.__create_map()

    def __create_map(self):
        self.room = Room(self.WIDTH, self.HEIGHT, self.FLOOR_COLOR)
        self.goal1 = Goal(
            self.GOAL_OFFSET + self.GOAL_SIZE // 2,
            self.HEIGHT // 2,
            self.GOAL_SIZE,
            self.GOAL_FALSE_COLOR,
        )

        self.goal2 = Goal(
            self.WIDTH - self.GOAL_OFFSET - self.GOAL_SIZE // 2,
            self.HEIGHT // 2,
            self.GOAL_SIZE,
            self.GOAL_FALSE_COLOR,
        )

        self.maps = pygame.sprite.Group(self.room, self.goal1, self.goal2)
        self.borders = pygame.sprite.Group(
            Border(
                (0, 0),
                self.WIDTH,
                Orientation.HORIZONTAL,
                self.BORDER_WIDTH,
                self.BORDER_COLOR,
            ),
            Border(
                (self.WIDTH, 0),
                self.HEIGHT,
                Orientation.VERTICAL,
                self.BORDER_WIDTH,
                self.BORDER_COLOR,
            ),
            Border(
                (0, self.HEIGHT),
                self.WIDTH,
                Orientation.HORIZONTAL,
                self.BORDER_WIDTH,
                self.BORDER_COLOR,
            ),
            Border(
                (0, 0),
                self.HEIGHT,
                Orientation.VERTICAL,
                self.BORDER_WIDTH,
                self.BORDER_COLOR,
            ),
        )

    def reset(self):
        self.channel = 0
        self.r_agent.reset(self.AGENT_POS)
        self.s_agent.reset()
        self.true_goal = random.randint(0, 1)

        if self.prev_true_goal == 1:
            self.true_goal = 0
            self.prev_true_goal = 0
            self.goal1.change_color(self.GOAL_TRUE_COLOR)
            self.goal2.change_color(self.GOAL_FALSE_COLOR)
        else:
            self.true_goal = 1
            self.prev_true_goal = 1
            self.goal1.change_color(self.GOAL_FALSE_COLOR)
            self.goal2.change_color(self.GOAL_TRUE_COLOR)

    def step(self):
        return

    def send_message(self, message: int):
        self.channel = message

    def get_message(self):
        return self.channel

    def draw(self, screen):
        self.maps.draw(screen)
        self.borders.draw(screen)
        self.players.draw(screen)

    def check_goal(self):
        if self.true_goal == 0:
            return self.r_agent.rect.colliderect(self.goal1.rect)
        else:
            return self.r_agent.rect.colliderect(self.goal2.rect)

    def check_collision(self):
        if self.true_goal == 0:
            return self.r_agent.rect.colliderect(self.goal2.rect)
        else:
            return self.r_agent.rect.colliderect(self.goal1.rect)

    def get_normalized_distance_from_goal(self):
        """
        エージェントとゴール間の正規化された距離
        """
        if self.true_goal == 0:
            goal_x = self.goal1.rect.center[0]
        else:
            goal_x = self.goal2.rect.center[0]

        return (goal_x - self.r_agent.pos.x) / self.WIDTH

    def get_normalized_distance_from_all_goals(self):
        """
        エージェントとゴール間の正規化された距離
        """

        return (
            np.array(
                [
                    self.goal1.rect.center[0] - self.r_agent.pos.x,
                    self.goal2.rect.center[0] - self.r_agent.pos.x,
                ]
            )
            / self.WIDTH
        )

    def get_normalized_agent_position(self):
        return np.array([self.r_agent.pos.x / self.WIDTH])

    def get_normalized_goal_position(self):
        if self.true_goal == 0:
            goal_x = self.goal1.rect.center[0]
        else:
            goal_x = self.goal2.rect.center[0]

        return np.array([goal_x / self.WIDTH])
