import random
import numpy as np
import pygame
from typing import List, Tuple

from envs.diamond.world import MuseumWorld


class DiamondEnv:
    """
    マルチエージェント2Dシミュレーション環境(エージェント数: 2)
    警備員に見つからずにダイアモンドを盗む

    Robot Agent: 周りの障害物をレーザーで観測しながら動くことができる
    Sensor Agent: 完全観測能力を持ち、Robot Agentにメッセージを送ることができる

    RAの観測
    - LiDARで観測した距離[0.0〜1.0] * レーザーの本数
    - エージェントから見たゴールの相対的な座標 (x, y)
    - SAからのメッセージ (One-Hot)

    SAの観測
    - RAの絶対位置 (x, y)
    - 警備員の絶対位置 (x, y)
    - ゴールの相対的な座標 (x, y)

    グローバル状態
    - LiDAR
    - RAの絶対位置 (x, y)
    - 警備員の絶対位置 (x, y)
    - ゴールの相対的な座標 (x, y)
    - SAからのメッセージ (One-Hot)
    """

    OFFSET = 40
    INFO_HEIGHT = 270
    INFO_MARGIN = 40
    SA_INFO_WIDTH = 400
    SA_INFO_MARGIN = 25

    BG_COLOR = (10, 16, 21)
    INFO_BG_COLOR = (30, 33, 36)
    INFO_TEXT_COLOR = (220, 220, 220)
    INFO_TEXT_COLOR2 = (145, 150, 155)

    FPS = 60

    FONT_NAME = "Arial"

    REWARD_SUCCESS = 100
    REWARD_FAILURE = -100
    REWARD_TIME_PENALTY = -0.01

    MAX_EPISODE_STEPS = 300

    def __init__(
        self,
        episode_limit: int,
        debug: bool,
        agent_velocity: float,
        guard_velocity: float,
        channel_size: int,
        lidar_angle: float,
        lidar_interval: float,
        reward_success: float,
        reward_failure: float,
        render: bool = False,
    ):
        pygame.init()
        pygame.display.set_caption("Diamond Env")
        np.set_printoptions(precision=2, suppress=True)

        self.episode_limit = episode_limit
        self.debug = debug
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        self.agent_velocity = agent_velocity
        self.guard_velocity = guard_velocity
        self.channel_size = channel_size
        self.lidar_angle = lidar_angle
        self.lidar_interval = lidar_interval

        self.world = MuseumWorld()
        self.n_agents = len(self.world.agents())

        num_lasers = self.world.get_num_lasers()

        # self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(
        #     low=-1, high=1, shape=(num_lasers + 2,), dtype=float
        # )

        self.window = None
        self.WINDOW_WIDTH = self.world.WIDTH + self.SA_INFO_WIDTH + 2 * self.OFFSET
        self.WINDOW_HEIGHT = self.world.HEIGHT + self.INFO_HEIGHT + self.OFFSET
        self.map_screen = pygame.Surface((self.world.WIDTH, self.world.HEIGHT))
        self.info_screen = pygame.Surface((self.WINDOW_WIDTH, self.INFO_HEIGHT))
        self.sa_screen = pygame.Surface((self.SA_INFO_WIDTH, self.world.HEIGHT))
        self.clock = pygame.time.Clock()

        self._episode = 0
        self._timestep = 0

        self.goal_reached = False
        self.failed = False

        self.success_count = 0

        self.goal_distance = 0
        self.laser_distances = []

        self.font1 = pygame.font.SysFont(self.FONT_NAME, 45)
        self.font2 = pygame.font.SysFont(self.FONT_NAME, 30)
        self.font3 = pygame.font.SysFont(self.FONT_NAME, 24)
        self.font4 = pygame.font.SysFont(self.FONT_NAME, 20)
        self.font5 = pygame.font.SysFont(self.FONT_NAME, 16)
        self.font6 = pygame.font.SysFont(self.FONT_NAME, 14)

    def get_env_info(self) -> dict:
        """
        環境の情報を取得する
        """
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_obs_size(self):
        """
        エージェントの部分観測のサイズを返す
        """
        # RAエージェントの観測サイズ
        # レーザー + ゴール相対座標 + メッセージ
        ra_obs_size = self.world.get_num_lasers() + 2 + self.channel_size

        # SAエージェントの観測サイズ
        # RA座標 + 警備員座標 + ゴール相対座標
        sa_obs_size = 2 + 2 + 2

        # 大きい方のサイズに合わせる
        return max(ra_obs_size, sa_obs_size)

    def get_state_size(self):
        """
        グローバル状態のサイズを返す
        """
        # レーザー + RA座標 + 警備員座標 + ゴール相対座標 + メッセージ
        return self.world.get_num_lasers() + 2 + 2 + 2 + self.channel_size

    def get_total_actions(self):
        """
        エージェントがとることのできる行動の数を返す
        """
        return self.n_actions

    def step(self, actions) -> Tuple[list, float, bool, bool, dict]:
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

        # assert len(actions) == self.n_agents
        self._timestep += 1

        self.world.step(actions)

        if self.world.check_collision():
            info["is_success"] = False
            self.failed = True
            terminated = True
        elif self.world.check_goal():
            info["is_success"] = True
            self.goal_reached = True
            terminated = True
            self.success_count += 1
        elif self._timestep >= self.MAX_EPISODE_STEPS:
            info["is_success"] = False
            info["TimeLimit.truncated"] = True
            terminated = True

        # ゴールまでの正規化された距離
        self.goal_distance = self.world.get_normalized_distance_from_goal()

        self.reward = self.__get_reward()
        self.observations = self.__get_observations()

        if self.render_mode == "human":
            self.render()

        return self.observations, self.reward, terminated, info

    def reset(self) -> list:
        """
        環境をリセットする

        戻り値:
            - observation: 観測値
        """
        self.world.reset(random_direction=True)
        self._timestep = 0
        self._episode += 1
        self.goal_reached = False
        self.failed = False
        self.goal_distance = 0
        self.laser_distances = []

        observation = self.__get_observations()
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
            # return self.REWARD_FAILURE
        else:
            return -1 * self.goal_distance
            # return self.REWARD_TIME_PENALTY

    def __get_observations(self) -> List[float]:
        """
        観測値を取得する
        """
        obs = self.world.get_observations()
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
        self.sa_screen.fill((self.BG_COLOR))

        self.world.draw(self.map_screen)
        self.__draw_info()

        self.window.blit(self.map_screen, (self.OFFSET, self.INFO_HEIGHT))
        self.window.blit(self.info_screen, (0, 0))
        self.window.blit(
            self.sa_screen, (self.OFFSET + self.world.WIDTH, self.INFO_HEIGHT)
        )

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
        ra_obs_text = self.font4.render(
            f"RA Obs: {self.observations[0]}",
            True,
            self.INFO_TEXT_COLOR2,
        )
        sa_obs_text = self.font4.render(
            f"RA Obs: {self.observations[1]}",
            True,
            self.INFO_TEXT_COLOR2,
        )
        reward_text = self.font3.render(
            f"Reward: {self.reward: .2f}",
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
        self.info_screen.blit(
            reward_text, (self.INFO_MARGIN + 24, self.INFO_MARGIN + 140)
        )

        pygame.draw.rect(
            self.sa_screen,
            self.INFO_BG_COLOR,
            (
                self.SA_INFO_MARGIN,
                0,
                self.SA_INFO_WIDTH - 2 * self.SA_INFO_MARGIN,
                self.world.HEIGHT,
            ),
            0,
            15,
        )
        sa_title = self.font3.render(
            "Sensor Agent",
            True,
            self.INFO_TEXT_COLOR,
        )
        sa_message_title = self.font4.render(
            "Send Message:",
            True,
            self.INFO_TEXT_COLOR2,
        )
        sa_message_value = self.font2.render(
            str(self.world.channel),
            True,
            self.INFO_TEXT_COLOR,
        )
        self.sa_screen.blit(sa_title, (self.SA_INFO_MARGIN + 15, 20))
        self.sa_screen.blit(sa_message_title, (self.SA_INFO_MARGIN + 15, 60))
        self.sa_screen.blit(sa_message_value, (self.SA_INFO_MARGIN + 100, 100))

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
