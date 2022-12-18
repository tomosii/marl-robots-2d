import math
import numpy as np
import pygame
from typing import List, Tuple

from envs.diamond.objects import Goal, Room, Wall, Border, Guard, Orientation
from envs.diamond.agents import RobotAgent, SensorAgent, Agent
from envs.diamond.utils import one_hot_encode


class World:
    def __init__(self):
        self.channel = None
        self.players: pygame.sprite.Group = None
        self.maps: pygame.sprite.Group = None
        self.walls: List[Wall] = None

    def agents(self) -> List[Agent]:
        raise NotImplementedError

    def step(self, agent: Agent, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def draw(self, screen: pygame.Surface):
        raise NotImplementedError

    def check_collision(self) -> bool:
        raise NotImplementedError

    def check_goal(self) -> bool:
        raise NotImplementedError


class MuseumWorld(World):
    AGENT_SIZE = 15
    GUARD_SIZE = 50
    GOAL_SIZE = 40

    CORRIDOR_WIDTH = 70
    BORDER_WIDTH = 2
    WALL_WIDTH = 170
    WALL_HEIGHT = 80
    WALL_GAP = 40
    NUM_WALLS = 3

    WALL_COLOR = (45, 50, 56)
    FLOOR_COLOR = (80, 85, 90)
    BORDER_COLOR = (60, 67, 75)
    GOAL_COLOR = (60, 65, 72)
    AGENT_COLOR = (70, 180, 180)
    GUARD_COLOR = (180, 60, 60)
    LASER_COLOR = (100, 105, 115)
    LASER_POINT_COLOR = (200, 60, 60)

    def __init__(
        self,
        agent_velocity: float,
        guard_velocity: float,
        channel_size: int,
        lidar_angle: float,
        lidar_interval: float,
    ):

        self.agent_velocity = agent_velocity
        self.guard_velocity = guard_velocity
        self.channel_size = channel_size
        self.lidar_angle = lidar_angle
        self.lidar_interval = lidar_interval

        self.WIDTH = 2 * self.CORRIDOR_WIDTH + self.WALL_WIDTH
        self.HEIGHT = (
            self.NUM_WALLS * self.WALL_HEIGHT
            + (self.NUM_WALLS - 1) * self.WALL_GAP
            + 2 * self.WALL_GAP
            + self.GOAL_SIZE
        )

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

        self.AGENT_POS = (
            self.WIDTH - self.CORRIDOR_WIDTH // 2,
            self.HEIGHT - self.WALL_HEIGHT // 2,
        )

        self.channel = None

        self.guard = Guard(
            self.GUARD_COLOR,
            self.GUARD_SIZE,
            self.guard_velocity,
        )
        self.players = pygame.sprite.Group(self.r_agent, self.guard)

        self.scan_points = []

        self.__create_map()

    def __create_map(self):
        self.room = Room(self.WIDTH, self.HEIGHT, self.FLOOR_COLOR)
        self.goal = Goal(
            self.WIDTH // 2,
            self.WALL_GAP + self.GOAL_SIZE // 2,
            self.GOAL_SIZE,
            self.GOAL_COLOR,
        )
        self.maps = pygame.sprite.Group(self.room, self.goal)
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

        for i in range(self.NUM_WALLS):
            wall_x = self.CORRIDOR_WIDTH
            wall_y = (
                self.HEIGHT - self.WALL_HEIGHT - i * (self.WALL_HEIGHT + self.WALL_GAP)
            )
            self.maps.add(
                Wall(
                    wall_x,
                    wall_y,
                    self.WALL_WIDTH,
                    self.WALL_HEIGHT,
                    self.WALL_COLOR,
                )
            )
            self.borders.add(
                Border(
                    (wall_x, wall_y),
                    self.WALL_WIDTH,
                    Orientation.HORIZONTAL,
                    self.BORDER_WIDTH,
                    self.BORDER_COLOR,
                ),
                Border(
                    (wall_x + self.WALL_WIDTH, wall_y),
                    self.WALL_HEIGHT,
                    Orientation.VERTICAL,
                    self.BORDER_WIDTH,
                    self.BORDER_COLOR,
                ),
                Border(
                    (wall_x, wall_y + self.WALL_HEIGHT),
                    self.WALL_WIDTH,
                    Orientation.HORIZONTAL,
                    self.BORDER_WIDTH,
                    self.BORDER_COLOR,
                ),
                Border(
                    (wall_x, wall_y),
                    self.WALL_HEIGHT,
                    Orientation.VERTICAL,
                    self.BORDER_WIDTH,
                    self.BORDER_COLOR,
                ),
            )
        img = pygame.image.load("src/envs/diamond/diamond.png")
        self.diamond_img = pygame.transform.scale(img, (self.GOAL_SIZE, self.GOAL_SIZE))

    def agents(self) -> List[Agent]:
        return [self.r_agent, self.s_agent]

    def reset(self, random_direction=False):
        self.channel = None
        self.r_agent.reset(self.AGENT_POS)
        self.s_agent.reset()

        self.guard.reset(
            (self.CORRIDOR_WIDTH // 2, self.HEIGHT - self.GUARD_SIZE // 2),
            random_direction=random_direction,
        )

    def step(self, actions):
        self.channel = 0

        self.r_agent.move(actions)
        return

        for agent, action in zip(self.agents(), actions):
            if agent.movable:
                agent.move(action)

            if agent.sendable:
                self.send_message(action)
        # self.guard.patrol()

    def send_message(self, action):
        self.channel = action

    def get_message(self):
        return self.channel

    def get_observations(self):

        # エージェントからみたゴールの正規化された相対的な座標 [x, y]
        relative_goal_position = self.get_relative_normalized_goal_position()

        # 正規化された距離のLiDARセンサー値 [d1, ..., dn]
        laser_distances = self.laser_scan()
        normalized_laser_distances = self.normalize_distances(laser_distances)

        # 2つを結合して観測とする [x, y, d1, ..., dn]
        return np.hstack((relative_goal_position, normalized_laser_distances))

        observations = []
        for agent in self.agents():
            if agent.movable:
                # エージェントからみたゴールの正規化された相対的な座標 [x, y]
                relative_goal_position = self.get_relative_normalized_goal_position()

                # 正規化された距離のLiDARセンサー値 [d1, ..., dn]
                laser_distances = self.laser_scan()
                normalized_laser_distances = self.normalize_distances(laser_distances)

                # 通信チャンネル
                message = self.get_message()
                one_hot_message = one_hot_encode(message, self.CHANNEL_SIZE)

                # 2つを結合して観測とする [x, y, d1, ..., dn]
                obs = np.hstack(
                    (
                        relative_goal_position,
                        normalized_laser_distances,
                        one_hot_message,
                    )
                )
                observations.append(obs)

            if agent.sendable:
                # goal_absolute_position = self.get_relative_normalized_goal_position()
                agent_absolute_position = self.get_agent_normalized_position()
                obstacle_absolute_position = (
                    self.get_moving_obstacle_normalized_position()
                )

                obs = np.hstack((agent_absolute_position, obstacle_absolute_position))
                observations.append(obs)

        return observations

    def draw(self, screen):
        self.maps.draw(screen)
        self.borders.draw(screen)
        self.__draw_lasers(screen)
        self.players.draw(screen)
        screen.blit(
            self.diamond_img, (self.WIDTH // 2 - self.GOAL_SIZE // 2, self.WALL_GAP)
        )

    def __draw_lasers(self, screen):
        for intersection in self.laser_points:
            pygame.draw.line(screen, self.LASER_COLOR, self.r_agent.pos, intersection)
            pygame.draw.circle(screen, self.LASER_POINT_COLOR, intersection, 2.5)

    def check_collision(self):
        return pygame.sprite.spritecollideany(self.r_agent, self.borders)

    def check_goal(self):
        return pygame.sprite.collide_rect(self.r_agent, self.goal)

    def __get_obstacle_lines(self) -> List:
        """
        レーザーとの交点を計算するために、すべての障害物の線分を取得する
        """
        lines = []
        for border in self.borders:
            lines.append((border.start, border.end))
        return lines

    def get_num_lasers(self) -> int:
        return self.LIDAR_ANGLE // self.LIDAR_INTERVAL

    def laser_scan(self) -> np.ndarray:
        self.laser_points = self.r_agent.laser_scan(self.__get_obstacle_lines())
        laser_distances = [
            math.sqrt(
                (point[0] - self.r_agent.pos.x) ** 2
                + (point[1] - self.r_agent.pos.y) ** 2
            )
            for point in self.laser_points
        ]
        return np.array(laser_distances)

    def normalize_distances(self, distances: np.ndarray) -> np.ndarray:
        return distances / self.LIDAR_RANGE

    def get_relative_normalized_goal_position(self) -> np.ndarray:
        return np.array(
            [
                (self.goal.rect.centerx - self.r_agent.pos.x) / self.WIDTH,
                (self.goal.rect.centery - self.r_agent.pos.y) / self.HEIGHT,
            ]
        )

    def get_distance_from_goal(self):
        """
        エージェントとゴール間の距離
        """
        return math.sqrt(
            (self.r_agent.pos.x - self.goal.rect.center[0]) ** 2
            + (self.r_agent.pos.y - self.goal.rect.center[1]) ** 2
        )

    def get_normalized_distance_from_goal(self) -> float:
        """
        エージェントとゴール間の正規化された距離
        """
        return self.get_distance_from_goal() / self.LIDAR_RANGE

    def get_agent_normalized_position(self):
        return np.array(
            [self.r_agent.pos.x / self.WIDTH, self.r_agent.pos.y / self.HEIGHT]
        )

    def get_moving_obstacle_normalized_position(self):
        return np.array([self.guard.pos.x / self.WIDTH, self.guard.pos.y / self.HEIGHT])
