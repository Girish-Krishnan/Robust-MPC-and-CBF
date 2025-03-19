from gym.envs.registration import register

register(
    id="Drone-v1",
    entry_point="gym_drone.envs.drone_v1:DroneEnv",
)

register(
    id="Drone-v2",
    entry_point="gym_drone.envs.drone_v2:DroneEnv",
)

register(
    id="Drone-v3",
    entry_point="gym_drone.envs.drone_v3:DroneEnv",
)

register(
    id="Drone-v4",
    entry_point="gym_drone.envs.drone_v4:DroneEnv",
)

register(
    id="Drone-v5",
    entry_point="gym_drone.envs.drone_v5:DroneEnv",
)