import gym
from stable_baselines3 import DQN

from envs.single.single_agent_gym import SingleAgentEnv


env = SingleAgentEnv()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000, log_interval=4)
