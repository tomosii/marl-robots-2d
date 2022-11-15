import random
from typing import List, Tuple
import pygame
import gym
from gym import spaces
from world import SimpleWorld


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

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        pygame.init()
        pygame.display.set_caption("Single Agent Environment")

        self.world = SimpleWorld()

        self.window = pygame.display.set_mode([self.WINDOW_WIDTH, self.WINDOW_HEIGHT])
        self.map_screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.info_screen = pygame.Surface((self.WINDOW_WIDTH, self.INFO_HEIGHT))
        self.clock = pygame.time.Clock()

        self.episode = 0
        self.timestep = 0

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

        self.world.step(action)
        if self.world.npc.pos == self.world.AGENT_POS:
            terminated = True
        elif self.world.check_collision():
            terminated = True

        self.timestep += 1

        if self.render_mode == "human":
            self.render()

        return None, None, terminated, False, {}

    def reset(self) -> list:
        """
        環境をリセットする

        戻り値:
            - observation: 観測値
        """
        pygame.display.flip()
        pygame.time.wait(100)
        self.world.reset()
        self.timestep = 0
        self.episode += 1

    def render(self, mode="human"):
        """
        環境を描画する
        """
        if self.render_mode == "human":
            # self.clock.tick(self.FPS)
            self.__draw()
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen)

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
        episode_text = self.font2.render(f"Episode: {0}", True, self.INFO_TEXT_COLOR)
        timestep_text = self.font3.render(
            f"Timestep: {self.timestep}", True, self.INFO_TEXT_COLOR2
        )
        distance_text = self.font3.render(
            f"Distance from goal: {0:.0f}",
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

    def close(self):
        """
        環境を閉じる
        """
        pygame.quit()


if __name__ == "__main__":
    episode_num = 10
    max_timestep = 500

    env = SingleAgentEnv()
    for episode in range(episode_num):
        env.reset()
        for timestep in range(max_timestep):
            action = random.randint(0, 3)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

    env.close()

    # if agent.check_goal(goal1):
    #     result_text = result_font.render("CLEAR !!", True, INFO_TEXT_COLOR)
    #     screen.blit(result_text, (400, 80))
    #     reset()
    #     episode += 1

    # if agent.collide_npc(npc) or agent.collide_walls(walls.sprites()):
    #     result_text = result_font.render("FAILED", True, INFO_TEXT_COLOR)
    #     screen.blit(result_text, (400, 80))
    #     reset()
    #     episode += 1
