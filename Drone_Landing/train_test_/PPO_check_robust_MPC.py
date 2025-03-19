import gym
import gym_drone
import mujoco
import glfw
import numpy as np
from stable_baselines3 import PPO

def main():
    from gym_drone.envs.drone_v5 import DroneEnv

    date = "0818"
    trial = "E"
    steps = "50000"  # Example checkpoint step name

    env = DroneEnv(
        xml_file="Drone_ver_1.0/drone-v1.xml",
        frame_skip=5,
        use_safety_filter=True,
        horizon=3,
        robust_noise_bound=0.05,
        collision_radius=0.35,
        landing_xy_threshold=0.2,
        max_acc=2.0
    )

    save_path = f'./save_model_{date}/{trial}/'
    model = PPO.load(save_path + f"Hexy_model_{date}{trial}_{steps}_steps")

    obs = env.reset()

    if not glfw.init():
        raise Exception("GLFW initialization failed!")

    window = glfw.create_window(800, 600, "Robust MPC Drone V5", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed!")
    glfw.make_context_current(window)

    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    option = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    cam.lookat[:] = np.array([0, 0, 1])
    cam.distance = 4.0
    cam.azimuth = 90
    cam.elevation = -20

    while not glfw.window_should_close(window):
        glfw.poll_events()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        mujoco.mj_step(env.model, env.data)

        mujoco.mjv_updateScene(env.model, env.data, option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewport = mujoco.MjrRect(0, 0, 800, 600)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)

        if done:
            obs = env.reset()

    glfw.terminate()

if __name__ == "__main__":
    main()
