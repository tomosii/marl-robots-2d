import numpy as np
import enum
import copy
import logging
import random


class GuessEnv:
    def __init__(self, seed, episode_limit):

        self.n_agents = 2

        self.n_actions = 2

        self.episode_limit = episode_limit

        self.guess_turn = False

        self.message = None

    def reset(self, episode, test_mode=False, print_log=False):
        """
        環境を初期化
        エージェントの観測とグローバル状態を返す
        """
        # タイムステップをリセット
        self._step_count = 0

        self.number = random.randint(0, 1)
        print("Number: ", self.number)

        self.guess_turn = False

        return self.get_obs(debug=False), self.get_state()

    def step(self, actions):

        print("Actions: ", actions)

        info = {"is_success": False}

        terminated = False

        if self._step_count >= self.episode_limit - 1:
            terminated = True

        if self.guess_turn:
            answer = actions[1]
            reward = self.get_reward(answer)
            self.guess_turn = False
            print("Guess: ", answer)
        else:
            if actions[0] == 1:
                self.message = 0
            elif actions[0] == 1:
                self.message = 1
            reward = 0
            self.guess_turn = True
            print("Send Message: ", self.message)

        if reward == 1:
            print("Success!")
            terminated = True
            info["is_success"] = True

        self._step_count += 1

        return reward, terminated, info

    def get_reward(self, answer):
        """
        報酬関数
        """

        if answer == self.number:
            reward = 1
        else:
            reward = -1

        return reward

    def get_env_info(self):
        """
        環境のパラメータ
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

    def get_obs_size(self):
        """
        部分観測のサイズを返す
        - 自身の各食品の満足度 (0.0~1.0)
        - 各食品の残量 (0.0~1.0)
        """
        return 2

    def get_state_size(self):
        """
        グローバル状態のサイズを返す
        """
        return 2

    def get_total_actions(self):
        """
        エージェントがとることのできる行動の数を返す
        """
        return self.n_actions

    def get_avail_actions(self):
        """
        全エージェントの選択可能な行動をリストで返す
        """
        if self.guess_turn:
            return np.array([[1, 1], [1, 1]])
        else:
            return np.array([[1, 1], [1, 1]])

    def get_obs(self, debug=True):
        """
        全てのエージェントの観測を1つのリストで返す
        - 各食品の残量 (0.0~1.0)
        - 自身の各食品の満足度 (0.0~1.0)
        NOTE: 分散実行時はエージェントは自分自身の観測のみ用いるようにする
        """
        # 各エージェントの観測を結合したものをグローバル状態とする
        num_obs = np.zeros(2)
        num_obs[self.number] = 1

        mes_obs = np.zeros(2)
        mes_obs[self.message] = 1

        self.obs = np.concatenate(([num_obs], [mes_obs]))

        return np.array([self.obs])

    def get_state(self):
        """
        グローバル状態を返す
        NOTE: この関数は分散実行時は用いないこと
        """
        # 各エージェントの観測を結合したものをグローバル状態とする
        state = np.zeros(2)
        state[self.number] = 1
        return np.array([state]).astype(np.float32)

    def close(self):
        return
