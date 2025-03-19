import gym
import gym_drone
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable

def lin_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate / clip range schedule.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

date = "0818"
trial = "C"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path='./save_model_'+date+'/'+trial,
    verbose=2,
    name_prefix='Hexy_model_'+date+trial
)

event_callback = EveryNTimesteps(
    n_steps=int(1e4),  # every n_steps, save the model
    callback=checkpoint_on_event
)

# Make a Vectorized environment that includes the safety filter
def make_safe_env():
    # Make a single environment with safety filter turned on
    env = gym.make("Drone-v3",
                   use_safety_filter=True,  # <--- Safety is "baked in" at training time
                   robust_noise_bound=0.05) # an example bound on state estimation error
    return env

env = make_vec_env(make_safe_env, n_envs=1)

model = PPO(
    "MlpPolicy",
    env=env,
    device='cuda',
    verbose=2,
    tensorboard_log='./hexy_tb_log_'+ date,
    learning_rate=lin_schedule(3e-4, 3e-6),
    clip_range=lin_schedule(0.3, 0.1),
    n_epochs=10,
    ent_coef=1e-4,
    batch_size=256*8,
    n_steps=256
)

model.learn(
    total_timesteps=20000000,
    callback=event_callback,  # every n_steps, save the model.
    tb_log_name='hexy_tb_'+date+trial,
    reset_num_timesteps=True, # use this if you want fresh counters
    progress_bar=True
)

model.save("Hexy_model")
del model

# Testing or continuing from saved model
model = PPO.load("Hexy_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
