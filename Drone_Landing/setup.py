from setuptools import setup, find_packages

setup(
    name="gym_drone",
    version="0.1",
    install_requires=["gym", "stable-baselines3", "mujoco"],  # Add dependencies here
    packages=find_packages(),
    include_package_data=True,
)
