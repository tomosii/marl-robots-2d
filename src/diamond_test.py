from envs.diamond.diamond import DiamondEnv
import random

if __name__ == "__main__":
    env = DiamondEnv(
        episode_limit=200,
        agent_velocity=3,
        guard_velocity=3,
        channel_size=1,
        lidar_angle=360,
        lidar_interval=90,
        reward_success=100,
        reward_failure=-100,
    )

    env.get_env_info()

    env.reset(episode=0)

    while True:
        action = random.randint(0, 3)
        print(env.get_avail_actions())
        action = random.randint(0, 3)
        reward, done, info = env.step([action, 5])
        obs = env.get_obs()
        print(obs)
        print(env.world.get_mileage())
        env.render()
        if done:
            env.reset(episode=0)
            print("done")
