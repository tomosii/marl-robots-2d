import random
import numpy as np
import pygame
import gym
from gym import spaces
from typing import List, Tuple

from envs.single.world import SimpleWorld


class SingleAgentEnv(gym.Env):
    """
    シングルエージェントのGym環境
    レーザーで周りを観測して障害物を避けながら、ゴールに向かう
    """

    metadata = {"render.modes": ["human"]}

    WIDTH = 600
    HEIGHT = 600

    OFFSET = 100
    INFO_HEIGHT = 220
    INFO_MARGIN = 40
    WINDOW_WIDTH = WIDTH + 2 * OFFSET
    WINDOW_HEIGHT = HEIGHT + OFFSET + INFO_HEIGHT

    BG_COLOR = (10, 16, 21)
    INFO_BG_COLOR = (30, 33, 36)
    INFO_TEXT_COLOR = (220, 220, 220)
    INFO_TEXT_COLOR2 = (145, 150, 155)

    FPS = 60

    FONT_NAME = "Arial"

    REWARD_SUCCESS = 100
    REWARD_FAILURE = -10
    REWARD_TIME_PENALTY = -0.1

    MAX_EPISODE_STEPS = 300

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        pygame.init()
        pygame.display.set_caption("Single Agent Environment")

        self.world = SimpleWorld()
        num_lasers = self.world.get_num_lasers()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(num_lasers + 2,), dtype=float
        )

        self.window = None
        self.map_screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.info_screen = pygame.Surface((self.WINDOW_WIDTH, self.INFO_HEIGHT))
        self.clock = pygame.time.Clock()

        self._episode = 0
        self._timestep = 0

        self.goal_reached = False
        self.failed = False

        self.goal_distance = 0
        self.laser_distances = []

        self.font1 = pygame.font.SysFont(self.FONT_NAME, 45)
        self.font2 = pygame.font.SysFont(self.FONT_NAME, 30)
        self.font3 = pygame.font.SysFont(self.FONT_NAME, 24)
        self.font4 = pygame.font.SysFont(self.FONT_NAME, 20)
        self.font5 = pygame.font.SysFont(self.FONT_NAME, 16)
        self.font6 = pygame.font.SysFont(self.FONT_NAME, 14)

    def step(self, action) -> Tuple[list, float, bool, bool, dict]:
        """
        行動を実行して、環境を1ステップ進める

        戻り値:
            - observation: 観測値
            - reward: 報酬
            - terminated: エピソード完了フラグ
            - truncated: エピソード中断フラグ
            - info: その他の情報
        """

        terminated = False
        info = {}

        self.world.step(action)

        if self.world.check_collision():
            self.failed = True
            terminated = True
        elif self.world.check_goal():
            info = {"is_success": True}
            self.goal_reached = True
            terminated = True
        elif self._timestep >= self.MAX_EPISODE_STEPS:
            info = {"TimeLimit.truncated": True}
            terminated = True

        self.goal_distance = self.world.get_normalized_distance_from_goal()

        reward = self.__get_reward()
        observation = self.__get_observation()

        if self.render_mode == "human":
            self.render()

        self._timestep += 1
        return observation, reward, terminated, info

    def reset(self) -> list:
        """
        環境をリセットする

        戻り値:
            - observation: 観測値
        """
        self.world.reset(random_direction=False)
        self._timestep = 0
        self._episode += 1
        self.goal_reached = False
        self.failed = False
        self.goal_distance = 0
        self.laser_distances = []

        observation = self.__get_observation()
        return observation

    def __get_reward(self) -> float:
        """
        報酬を計算する
        """
        if self.goal_reached:
            print("Goal reached!")
            return self.REWARD_SUCCESS
        elif self.failed:
            return self.REWARD_FAILURE - 1 * self.goal_distance
        else:
            return -1 * self.goal_distance

    def __get_observation(self) -> List[float]:
        """
        観測値を取得する
        """
        relative_goal_position = self.world.get_relative_normalized_goal_position()

        laser_distances = self.world.laser_scan()
        normalized_laser_distances = self.world.normalize_distances(laser_distances)

        obs = np.hstack((relative_goal_position, normalized_laser_distances))
        return obs

    def render(self, mode="human"):
        """
        環境を描画する
        """
        self.clock.tick(self.FPS)
        if self.window is None:
            self.window = pygame.display.set_mode(
                (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            )
        self.__draw()
        pygame.event.pump()
        pygame.display.flip()
        # elif self.render_mode == "rgb_array":
        #     return pygame.surfarray.array3d(self.screen)

    def __draw(self):
        """
        Pygameの描画処理
        """
        self.window.fill(self.BG_COLOR)
        self.map_screen.fill(self.BG_COLOR)
        self.info_screen.fill(self.BG_COLOR)

        self.world.draw(self.map_screen)
        self.__draw_info()

        self.window.blit(self.map_screen, (self.OFFSET, self.INFO_HEIGHT))
        self.window.blit(self.info_screen, (0, 0))

    def __draw_info(self):
        episode_text = self.font2.render(
            f"Episode: {self._episode}", True, self.INFO_TEXT_COLOR
        )
        timestep_text = self.font3.render(
            f"Timestep: {self._timestep}", True, self.INFO_TEXT_COLOR2
        )
        distance_text = self.font3.render(
            f"Distance from goal: {self.goal_distance:.3f}",
            True,
            self.INFO_TEXT_COLOR2,
        )
        pygame.draw.rect(
            self.info_screen,
            self.INFO_BG_COLOR,
            (
                self.INFO_MARGIN,
                self.INFO_MARGIN,
                self.WINDOW_WIDTH - 2 * self.INFO_MARGIN,
                self.INFO_HEIGHT - 2 * self.INFO_MARGIN,
            ),
            0,
            20,
        )
        self.info_screen.blit(
            episode_text, (self.INFO_MARGIN + 24, self.INFO_MARGIN + 20)
        )
        self.info_screen.blit(
            timestep_text, (self.INFO_MARGIN + 24, self.INFO_MARGIN + 65)
        )
        self.info_screen.blit(
            distance_text, (self.INFO_MARGIN + 24, self.INFO_MARGIN + 95)
        )

        if self.failed:
            result_text = self.font1.render("FAILED", True, (160, 50, 50))
            self.info_screen.blit(
                result_text, (self.WINDOW_WIDTH - 300, self.INFO_MARGIN + 50)
            )
        elif self.goal_reached:
            result_text = self.font1.render("CLEAR !!!!!!", True, (50, 160, 80))
            self.info_screen.blit(
                result_text, (self.WINDOW_WIDTH - 300, self.INFO_MARGIN + 50)
            )

    def close(self):
        """
        環境を閉じる
        """
        pygame.quit()


if __name__ == "__main__":
    episode_num = 10
    max_timestep = 100

    env = SingleAgentEnv()
    for episode in range(episode_num):
        env.reset()
        for timestep in range(max_timestep):
            action = random.randint(0, 3)
            observation, reward, terminated, truncated, info = env.step(1)

            if terminated:
                print(observation, reward)

                break

    env.close()
