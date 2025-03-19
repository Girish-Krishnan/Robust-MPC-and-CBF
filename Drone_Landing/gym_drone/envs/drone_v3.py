import numpy as np
import matplotlib.pyplot as plt
import mujoco
from gym import utils
from gym_drone.envs import mujoco_env
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)

DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}

class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
                 xml_file='Drone_ver_1.0/drone-v1.xml',
                 use_safety_filter=True,
                 robust_noise_bound=0.05):
        """
        :param xml_file: Path to the MuJoCo XML.
        :param use_safety_filter: If True, actions are filtered to ensure safety constraints.
        :param robust_noise_bound: Bound on state noise used to make the filter robust.
        """
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

        # Safety filter parameters
        self.use_safety_filter = use_safety_filter
        self.robust_noise_bound = robust_noise_bound  # how much position error we assume for robust safe flight

        # Body names
        self.car_body_array = ["Landing_box_col"]
        self.drone_body_array = ["Main_body_col_1", "Main_body_col_2", "Main_body_col_3", "Main_body_col_4"]
        self.drone_blade_array = ["FL_blade_col", "FR_blade_col", "BL_blade_col", "BR_blade_col"]

        # Initialize the Mujoco Environment
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def is_healthy(self):
        """
        A basic 'health' check that includes:
          - Drone vertical position not below a threshold
          - Drone pitch not too large
          - Distances not exceeding certain thresholds
          - No blade-body collisions with car
        """
        # Example constraints
        height_ok = (self.state_vector()[2] > -1.9)
        pitch_ok = (abs(self.state_vector()[4]) < 0.8)
        dist_ok = (self.dist_between_agents < 4.0 and self.xy_Distance_between_two_agents < 3)

        is_healthy = height_ok and pitch_ok and dist_ok

        # Check collisions: if any blade touches car's collision box, not healthy
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

    def apply_safety_filter(self, action):
        """
        Naive safety filter for robust collision avoidance.
        * We assume up to self.robust_noise_bound error in the drone's (x,y,z) position.
        * We do a rough check to see if next step might bring the blades into the car's bounding region.
        * If so, we modify the action to keep it safe, while still allowing approach from above.

        This is just an *example* of how to enforce a robust safety constraint.
        """

        # Current drone and car positions
        state = self.state_vector()  # [qpos, qvel] but we only use qpos for position
        x_drone = state[0] - 1.5
        y_drone = state[1]
        z_drone = state[2] + 2

        x_car = state[10] - 0.03
        y_car = state[11]
        z_car = state[12] + 0.1

        drone_pos = np.array([x_drone, y_drone, z_drone])
        car_pos = np.array([x_car, y_car, z_car])

        # Horizontal distance between drone and car
        xy_dist = np.linalg.norm(drone_pos[:2] - car_pos[:2])

        # For robust constraints, pretend the drone might be anywhere within robust_noise_bound
        # => effectively, we enlarge the "danger zone"
        effective_xy_dist = xy_dist - self.robust_noise_bound
        if effective_xy_dist < 0:
            effective_xy_dist = 0.0

        # Let's define a naive bounding radius for car + drone-blade.
        # If the drone is horizontally closer than "blade_radius" to the top of the car,
        # we consider it dangerous unless we are truly centered above the car so we can land.
        # This is just an example; your real approach might use a CBF or robust QP with partial dynamics.
        blade_radius = 0.35  # total bounding margin around the car for the blades, includes some extra margin.

        # If the effective distance < blade_radius, we only allow descending if we are near-centered above the car
        # (e.g., horizontal offset < 0.1) and we must approach from above (drone z > car z).
        # Otherwise, clamp the vertical action upward to 0 or a mild positive to avoid collision.
        # action = [thrust_x, thrust_y, thrust_z, torque_yaw]

        # We interpret action[2] as vertical thrust, with negative meaning descending. 
        # We'll clamp it if we're not well-centered and close.

        # Condition to check "centered above car"
        # (E.g., if you want to allow final approach only when horizontally aligned < some small threshold.)
        center_threshold = 0.1
        horizontal_offset = np.linalg.norm((drone_pos[:2] - car_pos[:2]))

        # The environment uses "action" as direct controls in x,y,z. 
        # This clamp is purely for demonstration:
        if effective_xy_dist < blade_radius:
            # If we are near the car horizontally but not well-centered, disallow descending
            if horizontal_offset > center_threshold:
                # If the raw action tries to push us down, clamp it to 0.
                if action[2] < 0.0:
                    # Force no descent
                    action[2] = 0.0
            # If we are centered above the car, we allow normal approach
            # but you might reduce the maximum speed of approach here, if desired
        else:
            # If we are not near the car, no special clamp. 
            # Still we might want to ensure we don't crash into ground. This is a minimal example.
            pass

        return action

    def step(self, action):
        """
        Single step in the environment.
        """

        # If the drone turned off after success, override action
        if self.turn_off_flag == 1:
            action = [0, 0, -0.1, 0]

        # ----------------------------------------------------
        # 1) Apply the safety filter if requested
        # ----------------------------------------------------
        if self.use_safety_filter:
            safe_action = self.apply_safety_filter(np.array(action, dtype=np.float32))
        else:
            safe_action = np.array(action, dtype=np.float32)

        # Save actions for potential penalty on changes
        self.action_buffer_2 = self.action_buffer
        self.action_buffer = safe_action
        self.input_history_buffer.append(safe_action)

        # ----------------------------------------------------
        # 2) Propagate the environment
        # ----------------------------------------------------
        self.data.ctrl[:] = safe_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Because qvel is forced in original code, we keep that:
        qpos = np.array(self.data.qpos)
        # qvel is partially derived from 'action' just to keep the environment controlled
        # This is the original hack in the environment
        qvel = np.array([
            safe_action[0],  # x thrust
            safe_action[1],  # y thrust
            safe_action[2],  # z thrust
            0,
            safe_action[3],  # yaw torque
            0, 0, 0, 0, 0,
            0.28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ])
        self.set_state(qpos, qvel)

        # ----------------------------------------------------
        # 3) Compute reward and done
        # ----------------------------------------------------
        # Positions
        x_drone = self.state_vector()[0] - 1.5
        y_drone = self.state_vector()[1]
        z_drone = self.state_vector()[2] + 2

        x_car = self.state_vector()[10] - 0.03
        y_car = self.state_vector()[11]
        z_car = self.state_vector()[12] + 0.1

        drone_pos = np.array([x_drone, y_drone, z_drone])
        car_pos = np.array([x_car, y_car, z_car])

        self.dist_between_agents = np.linalg.norm(drone_pos - car_pos)
        self.xy_Distance_between_two_agents = np.linalg.norm(drone_pos[:2] - car_pos[:2])

        # Distance-based reward
        dist_reward = 10 - (6 * self.xy_Distance_between_two_agents + 20 * abs(self.state_vector()[4]))

        land_reward = 0
        if self.xy_Distance_between_two_agents < 0.35:
            land_reward = 30 / (0.1 + np.abs(drone_pos[2] - car_pos[2]))
            print("Pole!")  # Debug print

        # Check if main body is fully touching
        touched_set = set()
        for i in range(self.data.ncon):
            sim_contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, sim_contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, sim_contact.geom2)

            if geom2 == self.car_body_array[0] and geom1 in self.drone_body_array:
                touched_set.add(geom1)

        goal_reward = 0
        if len(touched_set) == 4:
            goal_reward = 500000
            self.turn_off_flag = 1
            print("!!!!!!! SUCCESS !!!!!!! : !!!!!!! 4 Points are Touched !!!!!!!")

        # Blade collision penalty (already done in is_healthy, but we can do an additional reward penalty)
        col_reward = 0
        for i in range(self.data.ncon):
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom2)
            if geom2 == self.car_body_array[0] and geom1 in self.drone_blade_array:
                col_reward = -100
                break

        # Penalize large changes in action
        over_reward = -0.2 * np.linalg.norm(self.action_buffer_2 - self.action_buffer)

        reward = dist_reward + land_reward + goal_reward + col_reward + over_reward

        self.time_step += 1

        return self._get_obs(), reward, self.done, {'total reward': reward}

    def _get_obs(self):
        """
        Observation: 
        Action buffer (4) + pitch (1) + relative desired heading (3).
        """
        return np.concatenate([
            self.action_buffer,
            [self.state_vector()[4]],
            self.rel_desired_heading_vec
        ])

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
