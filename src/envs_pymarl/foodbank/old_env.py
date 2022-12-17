# Environmentクラス

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
import os

from agent import Agent
from status import StockRemaining, StockChange, Satisfaction, Progress
from config import LearningParameters as lp
from config import EnvironmentSettings as es


class Environment2:

    def __init__(self, f):
        self.f = f
        self.agents = self.init_agents()

    def init_agents(self):
        state_size = pow(len(StockRemaining), NUM_FOODS) * pow(len(StockChange),
                                                               NUM_FOODS) * pow(len(Satisfaction), NUM_FOODS) * len(Progress)
        num_states = NUM_FOODS * 3 + 1
        num_actions = NUM_FOODS + 1

        print("========= 各エージェントのもつ状態行動空間 =========", file=self.f)
        print(f"状態数: {state_size:,}", file=self.f)
        print(f"行動数: {num_actions}", file=self.f)
        print(f"状態 × 行動の組み合わせ: {(state_size * num_actions):,}", file=self.f)
        print("\n\n", file=self.f)

        agents: List[Agent] = []
        for i in range(AGENTS_COUNT):
            name = f"Agent{i + 1}"
            agent = Agent(
                name, np.array(REQUESTS[i]), num_states, num_actions, self.f)
            agents.append(agent)
        return agents

    def reset(self, greedy):
        self.stock = np.array(FOODS, dtype=np.int64)
        states = None

        for agent in self.agents:
            agent.reset(self.stock, greedy)

        return states

    def get_actions(self, states):
        actions = []
        # すべてのエージェントに対して
        for agent, state in zip(self.agents, states):
            # 行動を決定
            action = agent.get_action(state)
            actions.append(action)
            if action == NUM_FOODS:
                # print(f"{agent.name} 行動: 何もしない")
                pass
            else:
                # print(f"{agent.name} 行動: 食品{action}を１つ取る")
                pass
        return actions

    def check_food_run_out(self):
        # 全ての在庫が0になったかチェック
        return np.all(self.stock == 0)

    def check_agents_food_complete(self):
        # 全てのエージェントが終了条件を満たしているかチェック
        all_done = True
        for agent in self.agents:
            if not agent.check_food_complete(self.stock):
                all_done = False
                break
        # 全エージェントの取れる行動がなくなったか
        return all_done

    def check_agents_learning_done(self):
        all_done = True
        for agent in self.agents:
            if not agent.learning_done:
                all_done = False
                break
        return all_done

    # def learn(self, states, actions, rewards, states_next, alpha):
    #     for agent, state, action, reward, state_next in zip(self.agents, states, actions, rewards, states_next):
    #         agent.learn(state, action, reward, state_next, alpha)

    def get_reward(self, target_agent: Agent, terminal, greedy):
        if terminal:
            # - |制約違反度の平均からの偏差|
            # violations = []
            # for agent in self.agents:
            #     v = agent.get_violation()
            #     violations.append(v)
            # mean = np.mean(violations)
            # abs_deviation = np.absolute(mean - target_agent.get_violation())
            # reward = - abs_deviation

            # - (制約違反度 + 制約違反度の標準偏差)
            violations = []
            for agent in self.agents:
                v = agent.get_violation()
                # print(f"violation: {v}")
                violations.append(v)
            std = np.std(violations)
            print(f"std: {std}")

            reward = - (target_agent.get_violation() + std)
            reward = 0
            print(f"reward: {reward}")

            # - (制約違反度の平均+標準偏差)　統一
            # violations = []
            # for agent in self.agents:
            #     v = agent.get_violation()
            #     violations.append(v)
            # mean = np.mean(violations)
            # # std = np.std(violations)
            # # reward = - (mean + std)
            # reward = - mean

            if greedy:
                print(
                    f"{target_agent.name}: 報酬{reward:.3f}  要求{target_agent.REQUEST} 在庫{target_agent.stock}", file=self.f)

        else:
            reward = -1
            # reward = 0

        return reward

    def print_env_state(self):
        print("Env State: [", end="", file=self.f)
        for status in self.env_state:
            print(f"{status.name} ", end="", file=self.f)

        print("]", file=self.f)


class Environment:

    def __init__(self, f):
        self.f = f
        # state_size = pow(len(StockRemaining), es.NUM_FOODS) * \
        #     pow(len(Satisfaction), es.NUM_FOODS) * len(Progress)

        state_size = pow(len(StockRemaining), es.NUM_FOODS) * pow(len(StockChange),
                                                                  es.NUM_FOODS) * pow(len(Satisfaction), es.NUM_FOODS) * len(Progress)
        action_size = es.NUM_FOODS + 1

        print("========= 各エージェントのもつ状態行動空間 =========", file=self.f)
        print(f"状態数: {state_size:,}", file=self.f)
        print(f"行動数: {action_size}", file=self.f)
        print(f"状態 × 行動の組み合わせ: {(state_size * action_size):,}", file=self.f)
        print("\n\n", file=self.f)
        self.agents = self.init_agents()

    def init_agents(self):
        agents: List[Agent] = []
        for i in range(es.AGENTS_COUNT):
            name = f"Agent{i + 1}"
            agent = Agent(
                name, np.array(es.REQUESTS[i]), self.f)
            agents.append(agent)
        return agents

    def reset(self, greedy):
        self.stock = np.array(es.FOODS, dtype=np.int64)
        states = None

        for agent in self.agents:
            agent.reset(self.stock, greedy)

        return states

    def get_actions(self, states):
        actions = []
        # すべてのエージェントに対して
        for agent, state in zip(self.agents, states):
            # 行動を決定
            action = agent.decide_action(state)
            actions.append(action)
            if action == es.NUM_FOODS:
                # print(f"{agent.name} 行動: 何もしない")
                pass
            else:
                # print(f"{agent.name} 行動: 食品{action}を１つ取る")
                pass
        return actions

    def check_food_run_out(self):
        # 全ての在庫が0になったかチェック
        return np.all(self.stock == 0)

    def check_agents_food_done(self):
        # 全てのエージェントが終了条件を満たしているかチェック
        all_done = True
        for agent in self.agents:
            if not agent.food_done:
                all_done = False
                break
        # 全エージェントの取れる行動がなくなったか
        return all_done

    def check_agents_learning_done(self):
        all_done = True
        for agent in self.agents:
            if not agent.learning_done:
                all_done = False
                break
        return all_done

    def learn(self, states, actions, rewards, states_next, alpha):
        for agent, state, action, reward, state_next in zip(self.agents, states, actions, rewards, states_next):
            agent.learn(state, action, reward, state_next, alpha)

    def get_reward(self, target_agent: Agent, terminal, greedy):
        if terminal:
            # - |制約違反度の平均からの偏差|
            # violations = []
            # for agent in self.agents:
            #     v = agent.get_violation()
            #     violations.append(v)
            # mean = np.mean(violations)
            # abs_deviation = np.absolute(mean - target_agent.get_violation())
            # reward = - abs_deviation

            # - (制約違反度 + 制約違反度の標準偏差)
            violations = []
            for agent in self.agents:
                v = agent.get_violation()
                violations.append(v)
            std = np.std(violations)
            reward = - (target_agent.get_violation() + std)

            # - (制約違反度の平均+標準偏差)　統一
            # violations = []
            # for agent in self.agents:
            #     v = agent.get_violation()
            #     violations.append(v)
            # mean = np.mean(violations)
            # std = np.std(violations)
            # reward = - (mean + std)

            if greedy:
                # print(
                #     f"{target_agent.name}: 報酬{reward:.3f} 絶対偏差{abs_deviation:.1f} 満足度{target_satisfaction:.1f} 要求{target_agent.REQUEST} 在庫{target_agent.stock}", file=self.f)

                print(
                    f"{target_agent.name}: 報酬{reward:.3f}  要求{target_agent.REQUEST} 在庫{target_agent.stock}", file=self.f)

        else:
            reward = -1
            # reward = 0

        return reward

    def print_env_state(self):
        print("Env State: [", end="", file=self.f)
        for status in self.env_state:
            print(f"{status.name} ", end="", file=self.f)

        print("]", file=self.f)
