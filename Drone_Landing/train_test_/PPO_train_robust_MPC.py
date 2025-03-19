import gym
import gym_drone
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable

def linear_schedule(initial_value: float, final_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

date = "0818"
trial = "E"

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
    Creates an instance of the DroneEnvV5 with the robust MPC filter turned ON.
    """
    from gym_drone.envs.drone_v5 import DroneEnv
    env = DroneEnv(
        xml_file="Drone_ver_1.0/drone-v1.xml",
        frame_skip=5,
        use_safety_filter=True,
        horizon=3,                 # short horizon
        robust_noise_bound=0.05,   # bounding the noise
        collision_radius=0.35,
        landing_xy_threshold=0.2,
        max_acc=2.0
    )
    return env

if __name__ == "__main__":
    env = make_vec_env(make_drone_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env=env,
        device="cpu",  # or "cpu"
        verbose=2,
        tensorboard_log=f'./hexy_tb_log_{date}',
        learning_rate=linear_schedule(3e-4, 3e-6),
        clip_range=linear_schedule(0.3, 0.1),
        n_epochs=10,
        ent_coef=1e-4,
        batch_size=256 * 8,
        n_steps=256
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=event_callback,
        tb_log_name=f'hexy_tb_{date}{trial}',
        reset_num_timesteps=True,
        progress_bar=True
    )

    model.save(f"Hexy_model_{date}{trial}")
    del model

    # (Optional) test after training
    model = PPO.load(f"Hexy_model_{date}{trial}")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done.any():
            obs = env.reset()
