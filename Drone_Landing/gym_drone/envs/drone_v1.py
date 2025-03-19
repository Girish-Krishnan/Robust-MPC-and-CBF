import numpy as np
import matplotlib.pyplot as plt
import mujoco
from gym import utils
from gym_drone.envs import mujoco_env
from mpl_toolkits.mplot3d import Axes3D


DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}


class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file='Drone_ver_1.0/drone-v1.xml'):
        utils.EzPickle.__init__(**locals())

        #### Time_step
        self.time_step = 0

        #### For Drone
        self.xlist_drone = []
        self.ylist_drone = []
        self.zlist_drone = []

        #### For Car
        self.xlist_car = []
        self.ylist_car = []
        self.zlist_car = []

        ### Action buffer
        self.action_buffer = np.array([0, 0, 0, 0])
        self.action_buffer_2 = np.array([0, 0, 0, 0])

        ### Input History buffer
        self.input_history_buffer = []

        ## Distance between 2 agents
        self.dist_between_agents = 0
        self.xy_Distance_between_two_agents = 0

        ## Relative desired Position vector
        self.rel_desired_heading_vec = np.array([0, 0, 0])

        ## Turn OFF Flag
        self.turn_off_flag = 0

        # Body name for collision detection
        self.car_body_array = ["Landing_box_col"]
        self.drone_body_array = ["Main_body_col_1", "Main_body_col_2", "Main_body_col_3", "Main_body_col_4"]
        self.drone_blade_array = ["FL_blade_col", "FR_blade_col", "BL_blade_col", "BR_blade_col"]

        # Initialize the Mujoco Environment
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def is_healthy(self):
        is_healthy = self.state_vector()[2] > -1.9 and abs(self.state_vector()[4]) < 0.8
        is_healthy = is_healthy and self.dist_between_agents < 4.0 and self.xy_Distance_between_two_agents < 3

        for i in range(self.data.ncon):
            sim_contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, sim_contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, sim_contact.geom2)

            if geom2 == self.car_body_array[0] and geom1 in self.drone_blade_array:
                is_healthy = False
                print("Blade Collision!! : RESET")
                return is_healthy

        return is_healthy

    @property
    def done(self):
        return not self.is_healthy

    def step(self, action):
        if self.turn_off_flag == 1:
            action = [0, 0, -0.1, 0]

        self.action_buffer_2 = self.action_buffer
        self.action_buffer = action
        self.input_history_buffer.append(action)

        #### Run Simulation
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        qpos = np.array(self.data.qpos)
        qvel = np.array([action[0], action[1], action[2], 0, action[3], 0, 0, 0, 0, 0, 0.28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.set_state(qpos, qvel)

        #### Get Position INFO
        x_drone = self.state_vector()[0] - 1.5
        y_drone = self.state_vector()[1]
        z_drone = self.state_vector()[2] + 2

        x_car = self.state_vector()[10] - 0.03
        y_car = self.state_vector()[11]
        z_car = self.state_vector()[12] + 0.1

        drone_pos = np.array([x_drone, y_drone, z_drone])
        car_pos = np.array([x_car, y_car, z_car])

        #### Calculate Reward
        self.dist_between_agents = np.linalg.norm(drone_pos - car_pos)
        self.xy_Distance_between_two_agents = np.linalg.norm(drone_pos[:2] - car_pos[:2])

        dist_reward = 10 - (6 * self.xy_Distance_between_two_agents + 20 * abs(self.state_vector()[4]))

        land_reward = 0
        if self.xy_Distance_between_two_agents < 0.35:
            land_reward = 30 / (0.1 + np.abs(drone_pos[2] - car_pos[2]))
            print("Pole!")

        goal_reward = 0
        touched_set = set()
        for i in range(self.data.ncon):
            sim_contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, sim_contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, sim_contact.geom2)

            if geom2 == self.car_body_array[0] and geom1 in self.drone_body_array:
                touched_set.add(geom1)

        if len(touched_set) == 4:
            goal_reward = 500000
            self.turn_off_flag = 1
            print("!!!!!!! SUCCESS !!!!!!! : !!!!!!! 4 Points are Touched !!!!!!!")

        col_reward = -100 if any(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom1) in self.drone_blade_array and
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom2) == self.car_body_array[0]
            for i in range(self.data.ncon)
        ) else 0

        over_reward = -0.2 * np.linalg.norm(np.array(self.action_buffer_2) - np.array(self.action_buffer))

        reward = dist_reward + land_reward + goal_reward + col_reward + over_reward

        self.time_step += 1

        return self._get_obs(), reward, self.done, {'total reward': reward}

    def _get_obs(self):
        return np.concatenate([self.action_buffer, [self.state_vector()[4]], self.rel_desired_heading_vec])

    def reset_model(self):
        print("START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        mujoco.mj_resetData(self.model, self.data)

        observation = self._get_obs()

        self.time_step = 0
        self.xlist_drone.clear()
        self.ylist_drone.clear()
        self.zlist_drone.clear()

        self.xlist_car.clear()
        self.ylist_car.clear()
        self.zlist_car.clear()

        self.action_buffer = np.array([0, 0, 0, 0])
        self.action_buffer_2 = np.array([0, 0, 0, 0])

        self.turn_off_flag = 0
        self.input_history_buffer.clear()

        return observation

    def viewer_setup(self):
        renderer = mujoco.Renderer(self.model)
        renderer.update_scene(self.data)
        self.viewer = renderer
