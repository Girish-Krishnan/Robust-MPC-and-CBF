import gym
import gym_drone
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import cv2

def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func


date = "0815"
trial = "A"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path=f'./save_model_{date}/{trial}',
    verbose=2,
    name_prefix=f'Hexy_model_{date}{trial}'
)

event_callback = EveryNTimesteps(
    n_steps=int(1e4),  # every n_steps, save the model
    callback=checkpoint_on_event
)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        
        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # Ensure correct input channel size
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute the final flattened feature size dynamically
                test_input = th.zeros(1, *subspace.shape)  # Dummy input to get output shape
                with th.no_grad():
                    extracted_size = extractors[key](test_input).shape[1]
                
                total_concat_size += extracted_size

            elif key == "vector":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),  
                    nn.ReLU(),
                    nn.Linear(64, 32),  
                    nn.ReLU(),
                    nn.Linear(32, 16),  
                    nn.ReLU(),
                )
                total_concat_size += 16  # Final output size of vector branch

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size  # Update the expected feature dimension

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            # if image, save the image to a file
            # if key == "image":
            #     cv2.imwrite("image.png", cv2.cvtColor(255 * observations[key][0, :, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

        return th.cat(encoded_tensor_list, dim=1)  # Concatenate features correctly


env = make_vec_env("Drone-v2", n_envs=1)
policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)

model = PPO(
    "MultiInputPolicy", 
    env=env, 
    device="cuda", 
    policy_kwargs=policy_kwargs, 
    verbose=1,  
    tensorboard_log=f'./hexy_tb_log_{date}',
    learning_rate=lin_schedule(3e-4, 3e-6), 
    clip_range=lin_schedule(0.3, 0.1),
    n_epochs=10, 
    ent_coef=1e-4, 
    batch_size=256*4, 
    n_steps=1024
)

model.learn(
    total_timesteps=100000,
    callback=event_callback,  # every n_steps, save the model.
    tb_log_name=f'hexy_tb_{date}{trial}',
    progress_bar=True
)

model.save("Hexy_model")
del model  # remove to demonstrate saving and loading

model = PPO.load("Hexy_model")
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
