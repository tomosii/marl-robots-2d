import time
import random
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from envs.diamond.diamond_gym import DiamondGymEnv


TRAIN = True

env = DiamondGymEnv(
    episode_limit=100,
    agent_velocity=4,
    guard_velocity=3,
    channel_size=1,
    lidar_angle=360,
    lidar_interval=90,
    reward_success=100,
    reward_failure=-10,
)


if TRAIN:
    eval_env = DiamondGymEnv(
        episode_limit=100,
        agent_velocity=4,
        guard_velocity=3,
        channel_size=1,
        lidar_angle=360,
        lidar_interval=90,
        reward_success=100,
        reward_failure=-10,
        test_mode=True,
    )
    eval_callback = EvalCallback(
        eval_env, eval_freq=10000, n_eval_episodes=1, render=True
    )
    model = DQN("MlpPolicy", env, verbose=1, device="auto", exploration_fraction=0.9)
    model.learn(total_timesteps=300000, log_interval=50, callback=eval_callback)
    model.save("models/single_dqn")
    input(
        "Training finished. Press enter key to execute the trained agent. \n[PRESS ENTER]"
    )
    print("Success count: ", env.success_count)
else:
    model = DQN.load("../models/single_dqn", env=env)

# for episode in range(10):
#     obs = env.reset()
#     while True:
#         if random.random() < 0:
#             action = random.randint(0, 3)
#         else:
#             action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         env.render()
#         if done:
#             time.sleep(0.5)
#             break

env.close()
