import gym
import gym_drone  # Ensure your drone envs are registered
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import numpy as np


def linear_schedule(initial_value: float, final_value: float):
    """
    Linear schedule function for learning rate or clip range
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

date = "0818"
trial = "D"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path=f'./save_model_{date}/{trial}',
    verbose=2,
    name_prefix=f'Hexy_model_{date}{trial}'
)

event_callback = EveryNTimesteps(
    n_steps=int(1e4),
    callback=checkpoint_on_event
)

def make_drone_env():
    """
    Make an instance of the DroneEnvV4 with safety filter on.
    """
    from gym_drone.envs.drone_v4 import DroneEnv

    env = DroneEnv(
        xml_file="Drone_ver_1.0/drone-v1.xml",
        use_safety_filter=True,  # Train with safety on
        robust_noise_bound=0.05,
        collision_radius=0.35,
        landing_xy_threshold=0.15,
        cbf_relaxation=True
    )
    return env

if __name__ == "__main__":
    # Create a vectorized environment
    env = make_vec_env(make_drone_env, n_envs=1)

    # Build PPO model
    model = PPO(
        "MlpPolicy",
        env=env,
        device='cuda',   # or 'cpu'
        verbose=2,
        tensorboard_log=f'./hexy_tb_log_{date}',
        learning_rate=linear_schedule(3e-4, 3e-6),
        clip_range=linear_schedule(0.3, 0.1),
        n_epochs=10,
        ent_coef=1e-4,
        batch_size=256 * 8,
        n_steps=256
    )

    # Learn
    model.learn(
        total_timesteps=2_000_000,  # set for demonstration
        callback=event_callback,
        tb_log_name=f'hexy_tb_{date}{trial}',
        reset_num_timesteps=True,
        progress_bar=True
    )

    # Save final
    model.save(f"Hexy_model_{date}{trial}")
    del model

    # (Optional) Evaluate or reload
    model = PPO.load(f"Hexy_model_{date}{trial}")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done.any():
            obs = env.reset()
