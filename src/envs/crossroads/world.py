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


class TwoCrossroadsWorld(World):
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
    LASER_COLOR = (100, 105, 115)

    AGENT_COLOR = (70, 180, 180)

    def __init__(
        self,
        agent_velocity: float,
        channel_size: int,
        lidar_angle: float,
        lidar_interval: float,
    ):

        self.agent_velocity = agent_velocity
        self.channel_size = channel_size

        self.lidar_angle = lidar_angle
        self.lidar_interval = lidar_interval

        self.WIDTH = self.ROAD_WIDTH
        self.HEIGHT = self.ROAD_HEIGHT

        self.lidar_range = math.sqrt(self.WIDTH**2 + self.HEIGHT**2)


        self.r_agent = RobotAgent(
            self.AGENT_COLOR,
            self.AGENT_SIZE,
            self.agent_velocity,
            self.lidar_range,
            self.lidar_angle,
            self.lidar_interval,
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
        self.goals = [self.goal1, self.goal2]

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
        self.channel = None
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
        self.__draw_lasers(screen)
        self.players.draw(screen)

    def check_goal(self):
        if self.true_goal == 0:
            return self.r_agent.rect.colliderect(self.goal1.rect)
        else:
            return self.r_agent.rect.colliderect(self.goal2.rect)

    def check_collision(self):
        if pygame.sprite.spritecollideany(self.r_agent, self.borders) is not None:
            return True
            
        if self.true_goal == 0:
            return self.r_agent.rect.colliderect(self.goal2.rect)
        else:
            return self.r_agent.rect.colliderect(self.goal1.rect)

    def __draw_lasers(self, screen):
        for intersection in self.laser_points:
            pygame.draw.line(screen, self.LASER_COLOR, self.r_agent.pos, intersection)
            pygame.draw.circle(screen, self.AGENT_COLOR, intersection, 2.5)

    def __get_obstacle_lines(self) -> List:
        """
        レーザーとの交点を計算するために、すべての障害物の線分を取得する
        """
        lines = []
        for border in self.borders:
            lines.append((border.start, border.end))
        return lines

    def get_num_lasers(self) -> int:
        return self.lidar_angle // self.lidar_interval

    def laser_scan(self) -> np.ndarray:
        self.laser_points = self.r_agent.laser_scan(self.__get_obstacle_lines())
        self.laser_distances = np.array([
            math.sqrt(
                (point[0] - self.r_agent.pos.x) ** 2
                + (point[1] - self.r_agent.pos.y) ** 2
            )
            for point in self.laser_points
        ]) / self.lidar_range
        return self.laser_distances

    def get_relative_goal_position(self, goal: Goal):
        return np.array(
            [
                (goal.rect.centerx - self.r_agent.pos.x) / self.WIDTH,
                (goal.rect.centery - self.r_agent.pos.y) / self.HEIGHT,
            ]
        )

    def get_relative_goals_positions(self):

        positions = []
        for goal in self.goals:
            positions.append(
                self.get_relative_goal_position(goal)
            )

        return np.concatenate(positions)

    def get_relative_true_goal_position(self):
        if self.true_goal == 0:
            return self.get_relative_goal_position(self.goal1)
        else:
            return self.get_relative_goal_position(self.goal2)


    def get_normalized_agent_position(self):
        return np.array([self.r_agent.pos.x / self.WIDTH])

    def get_normalized_goal_position(self):
        if self.true_goal == 0:
            goal_x = self.goal1.rect.center[0]
        else:
            goal_x = self.goal2.rect.center[0]

        return np.array([goal_x / self.WIDTH])

    def get_distance_from_goal(self):
        if self.true_goal == 0:
            goal_x = self.goal1.rect.center[0]
            goal_y = self.goal1.rect.center[1]
        else:
            goal_x = self.goal2.rect.center[0]
            goal_y = self.goal2.rect.center[1]

        return math.sqrt(
            (goal_x - self.r_agent.pos.x) ** 2
            + (goal_y - self.r_agent.pos.y) ** 2
        ) / self.lidar_range


# class FourCrossroadsWorld(World):
#     AGENT_SIZE = 30
#     GOAL_SIZE = 50

#     ROAD_WIDTH = 500
#     ROAD_HEIGHT = 150
#     GOAL_OFFSET = 40

#     BORDER_WIDTH = 2

#     WALL_COLOR = (45, 50, 56)
#     FLOOR_COLOR = (80, 85, 90)
#     BORDER_COLOR = (60, 67, 75)
#     GOAL_TRUE_COLOR = (150, 65, 72)
#     GOAL_FALSE_COLOR = (60, 65, 72)

#     AGENT_COLOR = (70, 180, 180)

#     def __init__(
#         self,
#         agent_velocity: float,
#         channel_size: int,
#         num_goals: int,
#         lidar_angle: float,
#         lidar_interval: float,
#     ):

#         self.agent_velocity = agent_velocity
#         self.channel_size = channel_size
#         self.num_goals = num_goals

#         self.lidar_angle = lidar_angle
#         self.lidar_interval = lidar_interval

#         self.WIDTH = self.ROAD_WIDTH
#         self.HEIGHT = self.ROAD_HEIGHT

#         self.r_agent = RobotAgent(
#             self.AGENT_COLOR,
#             self.AGENT_SIZE,
#             self.agent_velocity,
#         )
#         self.s_agent = SensorAgent()
#         self.agents: List[Agent] = [self.r_agent, self.s_agent]

#         self.AGENT_POS = (
#             self.WIDTH // 2,
#             self.HEIGHT // 2,
#         )

#         self.channel = None

#         self.true_goal = 0
#         self.prev_true_goal = 0

#         self.players = pygame.sprite.Group(self.r_agent)

#         self.__create_map()

#     def __create_map(self):

#         assert self.num_goals in [2, 4], "num_goals must be 2 or 4"

#         self.borders = pygame.sprite.Group()
#         self.goals = pygame.sprite.Group()
        
#         if self.num_goals == 2:
#             self.goals.add(
#                 Goal(
#                     self.GOAL_OFFSET + self.GOAL_SIZE // 2,
#                     self.HEIGHT // 2,
#                     self.GOAL_SIZE,
#                     self.GOAL_FALSE_COLOR,
#                 ),
#                 Goal(
#                     self.WIDTH - self.GOAL_OFFSET - self.GOAL_SIZE // 2,
#                     self.HEIGHT // 2,
#                     self.GOAL_SIZE,
#                     self.GOAL_FALSE_COLOR,
#                 ),
#             )
#         elif self.num_goals == 4:
#             pass

#         self.room = Room(self.WIDTH, self.HEIGHT, self.FLOOR_COLOR)

#         self.maps = pygame.sprite.Group(self.room, self.goals)

#         self.borders.add(
#             Border(
#                 (0, 0),
#                 self.WIDTH,
#                 Orientation.HORIZONTAL,
#                 self.BORDER_WIDTH,
#                 self.BORDER_COLOR,
#             ),
#             Border(
#                 (self.WIDTH, 0),
#                 self.HEIGHT,
#                 Orientation.VERTICAL,
#                 self.BORDER_WIDTH,
#                 self.BORDER_COLOR,
#             ),
#             Border(
#                 (0, self.HEIGHT),
#                 self.WIDTH,
#                 Orientation.HORIZONTAL,
#                 self.BORDER_WIDTH,
#                 self.BORDER_COLOR,
#             ),
#             Border(
#                 (0, 0),
#                 self.HEIGHT,
#                 Orientation.VERTICAL,
#                 self.BORDER_WIDTH,
#                 self.BORDER_COLOR,
#             ),
#         )

#     def reset(self):
#         self.channel = None
#         self.r_agent.reset(self.AGENT_POS)
#         self.s_agent.reset()
#         self.true_goal = random.randint(0, 1)

#         if self.prev_true_goal == 1:
#             self.true_goal = 0
#             self.prev_true_goal = 0
#             self.goal1.change_color(self.GOAL_TRUE_COLOR)
#             self.goal2.change_color(self.GOAL_FALSE_COLOR)
#         else:
#             self.true_goal = 1
#             self.prev_true_goal = 1
#             self.goal1.change_color(self.GOAL_FALSE_COLOR)
#             self.goal2.change_color(self.GOAL_TRUE_COLOR)

#     def step(self):
#         return

#     def send_message(self, message: int):
#         self.channel = message

#     def get_message(self):
#         return self.channel

#     def draw(self, screen):
#         self.maps.draw(screen)
#         self.borders.draw(screen)
#         self.__draw_lasers(screen)
#         self.players.draw(screen)

#     def check_goal(self):
#         if self.true_goal == 0:
#             return self.r_agent.rect.colliderect(self.goal1.rect)
#         else:
#             return self.r_agent.rect.colliderect(self.goal2.rect)

#     def check_collision(self):
#         if self.true_goal == 0:
#             return self.r_agent.rect.colliderect(self.goal2.rect)
#         else:
#             return self.r_agent.rect.colliderect(self.goal1.rect)

#     def __draw_lasers(self, screen):
#         for intersection in self.laser_points:
#             pygame.draw.line(screen, self.LASER_COLOR, self.r_agent.pos, intersection)
#             pygame.draw.circle(screen, self.LASER_POINT_COLOR, intersection, 2.5)

#     def check_collision(self):
#         return pygame.sprite.spritecollideany(self.r_agent, self.borders)

#     def __get_obstacle_lines(self) -> List:
#         """
#         レーザーとの交点を計算するために、すべての障害物の線分を取得する
#         """
#         lines = []
#         for border in self.borders:
#             lines.append((border.start, border.end))
#         return lines

#     def get_num_lasers(self) -> int:
#         return self.lidar_angle // self.lidar_interval

#     def get_normalized_distance_from_goal(self):
#         """
#         エージェントとゴール間の正規化された距離
#         """
#         if self.true_goal == 0:
#             goal_x = self.goal1.rect.center[0]
#         else:
#             goal_x = self.goal2.rect.center[0]

#         return (goal_x - self.r_agent.pos.x) / self.WIDTH

#     def get_normalized_distance_from_all_goals(self):
#         """
#         エージェントとゴール間の正規化された距離
#         """

#         return (
#             np.array(
#                 [
#                     self.goal1.rect.center[0] - self.r_agent.pos.x,
#                     self.goal2.rect.center[0] - self.r_agent.pos.x,
#                 ]
#             )
#             / self.WIDTH
#         )

#     def get_normalized_agent_position(self):
#         return np.array([self.r_agent.pos.x / self.WIDTH])

#     def get_normalized_goal_position(self):
#         if self.true_goal == 0:
#             goal_x = self.goal1.rect.center[0]
#         else:
#             goal_x = self.goal2.rect.center[0]

#         return np.array([goal_x / self.WIDTH])
