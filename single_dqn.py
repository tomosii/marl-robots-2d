from stable_baselines3 import DQN
from envs.single.single_agent_gym import SingleAgentEnv

import pygame

TRAIN = True

env = SingleAgentEnv()

if TRAIN:
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
    )
    model.learn(total_timesteps=250000, log_interval=50)
    model.save("single_dqn")
    input("Training finished. Press any key to execute the trained agent.")
else:
    model = DQN.load("single_dqn", env=env)

for episode in range(5):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            pygame.time.wait(1000)
            break

env.close()
