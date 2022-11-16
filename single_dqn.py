import time
import random
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from envs.single.single_agent_gym import SingleAgentEnv

import pygame

TRAIN = True

env = SingleAgentEnv()


if TRAIN:
    eval_env = SingleAgentEnv()
    eval_callback = EvalCallback(eval_env, eval_freq=50000, render=True)
    model = DQN("MlpPolicy", env, verbose=1, device="auto")
    model.learn(total_timesteps=300000, log_interval=50, callback=eval_callback)
    model.save("single_dqn")
    input(
        "Training finished. Press enter key to execute the trained agent. \n[PRESS ENTER]"
    )
else:
    model = DQN.load("single_dqn", env=env)

for episode in range(10):
    obs = env.reset()
    while True:
        if random.random() < 0:
            action = random.randint(0, 3)
        else:
            action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            time.sleep(0.5)
            break

env.close()
