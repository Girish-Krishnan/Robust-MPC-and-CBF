import gym
import gym_drone
import mujoco
import glfw
import numpy as np
from stable_baselines3 import PPO
import time

def main():
    from gym_drone.envs.drone_v4 import DroneEnv

    date = "0818"
    trial = "D"
    steps = "1900000"  # example checkpoint step name

    # Create environment with safety
    env = DroneEnv(
        xml_file="Drone_ver_1.0/drone-v1.xml",
        use_safety_filter=True,  # Maintain safety in inference
        robust_noise_bound=0.05,
        collision_radius=0.35,
        landing_xy_threshold=0.15,
        cbf_relaxation=True
    )

    # Load model
    save_path = f'./save_model_{date}/{trial}/'
    model = PPO.load(save_path + f"Hexy_model_{date}{trial}_{steps}_steps")

    obs = env.reset()

    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW initialization failed!")

    # Make a window
    window = glfw.create_window(500, 500, "CBF-QP Drone V4", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed!")
    glfw.make_context_current(window)

    # Create the MuJoCo scene/context
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    option = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    # Example camera setup
    cam.lookat[:] = np.array([0, 0, 1])
    cam.distance = 1.0
    cam.azimuth = 0
    cam.elevation = 0
    mujoco.mjv_defaultOption(option)

    # Run loop
    started = False
    total_count = 0
    success_count = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # PPO inference
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # print("Reward: ", reward, "Done: ", done)

        # Step simulation
        mujoco.mj_step(env.model, env.data)

        # Update scene
        mujoco.mjv_updateScene(
            env.model, env.data, option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        # Render
        viewport = mujoco.MjrRect(0, 0, 800, 800)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)

        if done:
            obs = env.reset()
            total_count += 1

        if reward > 500_000:
            if not started:
                start_time = time.time()
                started = True
            
            else:
                if time.time() - start_time > 5:
                    obs = env.reset()
                    success_count += 1
                    total_count += 1
                    started = False


    glfw.terminate()
    print(f"Success rate: {success_count}/{total_count}")

if __name__ == "__main__":
    main()
