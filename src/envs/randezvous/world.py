import math
import numpy as np
import pygame
from typing import List, Tuple

from envs.randezvous.objects import Goal, Room, Wall, Border, Guard, Orientation
from envs.randezvous.agents import RobotAgent, SensorAgent, Agent
from envs.randezvous.utils import one_hot_encode


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


class RandezvousWorld(World):
    AGENT_SIZE = 15
    GUARD_SIZE = 50
    GOAL_SIZE = 50

    CORRIDOR_WIDTH = 70
    BORDER_WIDTH = 2
    WALL_WIDTH = 170
    WALL_HEIGHT = 45
    WALL_GAP = 50
    NUM_WALLS = 1

    WALL_COLOR = (45, 50, 56)
    FLOOR_COLOR = (80, 85, 90)
    BORDER_COLOR = (60, 67, 75)
    GOAL_COLOR = (60, 65, 72)
    AGENT1_COLOR = (70, 180, 180)
    AGENT2_COLOR = (180, 60, 60)
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

        agent1 = RobotAgent(
            self.AGENT1_COLOR,
            self.AGENT_SIZE,
            self.agent_velocity,
            self.lidar_range,
            self.lidar_angle,
            self.lidar_interval,
        )

        agent2 = RobotAgent(
            self.AGENT2_COLOR,
            self.AGENT_SIZE,
            self.agent_velocity,
            self.lidar_range,
            self.lidar_angle,
            self.lidar_interval,
        )
        self.agents = [agent1, agent2]

        self.initial_positions = [
            (
                self.WIDTH - self.CORRIDOR_WIDTH // 2,
                self.HEIGHT - self.WALL_HEIGHT // 2,
            ),
            (self.CORRIDOR_WIDTH // 2, self.HEIGHT - self.WALL_HEIGHT // 2),
        ]

        # self.AGENT1_POS = (
        #     self.WIDTH - self.CORRIDOR_WIDTH // 2,
        #     self.HEIGHT - self.WALL_HEIGHT // 2,
        # )

        # self.AGENT2_POS = (
        #     self.CORRIDOR_WIDTH // 2,
        #     self.HEIGHT - self.WALL_HEIGHT // 2,
        # )

        self.channel = None

        self.players = pygame.sprite.Group(agent1, agent2)

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

    def reset(self, random_direction=False):
        self.channel = None

        for i, agent in enumerate(self.agents):
            agent.reset(self.initial_positions[i])

        # self.guard.reset(
        #     (self.CORRIDOR_WIDTH // 2, self.HEIGHT - self.GUARD_SIZE // 2),
        #     random_direction=random_direction,
        # )

    def step(self):
        self.channel = 0
        # self.guard.patrol()

    def send_message(self, message: int):
        self.channel = message

    def get_message(self):
        return self.channel

    def draw(self, screen):
        self.maps.draw(screen)
        self.borders.draw(screen)
        self.__draw_lasers(screen)
        self.players.draw(screen)
        screen.blit(
            self.diamond_img, (self.WIDTH // 2 - self.GOAL_SIZE // 2, self.WALL_GAP)
        )

    def __draw_lasers(self, screen):
        for agent in self.agents:
            for intersection in agent.laser_points:
                pygame.draw.line(screen, agent.color, agent.pos, intersection)
                pygame.draw.circle(screen, self.LASER_POINT_COLOR, intersection, 2.5)

    def check_collision(self):
        for agent in self.agents:
            if pygame.sprite.spritecollideany(agent, self.borders):
                return True

    def check_both_goal(self):
        for agent in self.agents:
            if not pygame.sprite.collide_rect(self.goal, agent):
                return False
        return True

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

    def laser_scan(self, agent: RobotAgent) -> np.ndarray:
        obstacle_lines = self.__get_obstacle_lines()
        laser_points = agent.laser_scan(obstacle_lines)
        laser_distances = [
            math.sqrt((point[0] - agent.pos.x) ** 2 + (point[1] - agent.pos.y) ** 2)
            for point in laser_points
        ]
        # 正規化
        return laser_distances / self.lidar_range

    def get_relative_normalized_goal_position(self, agent: RobotAgent) -> np.ndarray:
        return np.array(
            [
                (self.goal.rect.centerx - agent.pos.x) / self.WIDTH,
                (self.goal.rect.centery - agent.pos.y) / self.HEIGHT,
            ]
        )

    def get_normalized_agent_position(self, agent: RobotAgent):
        return np.array([agent.pos.x / self.WIDTH, agent.pos.y / self.HEIGHT])

    # def get_normalized_guard_position(self):
    #     return np.array([self.guard.pos.x / self.WIDTH, self.guard.pos.y / self.HEIGHT])

    # def get_mileage(self):
    #     return self.r_agent.mileage

    def get_sum_distance_from_goal(self):
        return sum(
            [
                math.sqrt(
                    (agent.pos.x - self.goal.rect.center[0]) ** 2
                    + (agent.pos.y - self.goal.rect.center[1]) ** 2
                )
                / self.lidar_range
                for agent in self.agents
            ]
        )
