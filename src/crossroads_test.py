from envs.crossroads.crossroads import CrossroadsEnv
import random

if __name__ == "__main__":
    env = CrossroadsEnv(
        episode_limit=200,
        agent_velocity=5,
        channel_size=2,
        lidar_angle=360,
        lidar_interval=90,
        reward_success=1,
        reward_failure=-1,
        reward_step=0,
        n_goals=2,

    )

    env.get_env_info()

    env.reset(episode=0)

    while True:
        action = random.randint(0, 3)
        reward, done, info = env.step([3, 5])
        obs = env.get_obs()
        print(obs)
        env.render()
        if done:
            env.reset(episode=0)
            print("done")
