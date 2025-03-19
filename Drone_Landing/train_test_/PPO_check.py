import gym
import gym_drone
import mujoco
import glfw
import numpy as np
from stable_baselines3 import PPO

# -------------------------------
# Configuration
# -------------------------------
date = "0818"
trial = "A"
steps = "200000"

# -------------------------------
# Create Gym Environment
# -------------------------------
env = gym.make("Drone-v1")

# -------------------------------
# Load Trained Model
# -------------------------------
save_path = f'./save_model_{date}/{trial}/'
model = PPO.load(save_path + f"Hexy_model_{date}{trial}_{steps}_steps")

# Reset environment
obs = env.reset()

# -------------------------------
# Initialize MuJoCo Viewer Window
# -------------------------------
if not glfw.init():
    raise Exception("GLFW initialization failed!")

# Get MuJoCo model and data from the environment
model_mj = env.model
data = env.data

# Create a GLFW window
window = glfw.create_window(500, 500, "MuJoCo Drone Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed!")

glfw.make_context_current(window)

# Create MuJoCo scene and context
scene = mujoco.MjvScene(model_mj, maxgeom=1000)
context = mujoco.MjrContext(model_mj, mujoco.mjtFontScale.mjFONTSCALE_150)

# Create visualization options
opt = mujoco.MjvOption()

# Camera initialization
# Initialize MuJoCo camera
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

# Set camera position (xyz coordinates)
cam.lookat[:] = np.array([0, 0, 1])  # Drone's position
cam.distance = 0.5  # Close-up view
cam.azimuth = 0  # Rotate behind the drone
cam.elevation = 0  # Level with the horizon


mujoco.mjv_defaultOption(opt)

# -------------------------------
# Run Simulation Loop
# -------------------------------
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Get model prediction
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    # Step the simulation
    mujoco.mj_step(model_mj, data)

    # Update scene
    mujoco.mjv_updateScene(model_mj, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

    # Render the scene
    viewport = mujoco.MjrRect(0, 0, 1200, 900)
    mujoco.mjr_render(viewport, scene, context)

    # Swap buffers to display the updated frame
    glfw.swap_buffers(window)

    # Reset environment if episode is done
    if done:
        obs = env.reset()

# -------------------------------
# Cleanup
# -------------------------------
glfw.terminate()
