from stable_baselines3 import DQN

from stable_baselines3.common.env_checker import check_env
from envs.single.single_agent_gym import SingleAgentEnv


env = SingleAgentEnv()

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
)
model.learn(total_timesteps=200000, log_interval=4)

model.save("single_dqn")

input("Training finished. Press any key to execute the trained agent.")


for episode in range(5):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break
