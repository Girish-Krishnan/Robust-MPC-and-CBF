import numpy as np
import mujoco
import cvxpy as cp
from gym import utils
from gym_drone.envs import mujoco_env


class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Drone Environment using a simplified CBF-based QP for robust collision avoidance.
    We rely on a simple 6D state [x, y, z, vx, vy, vz] + ... and a simple single-constraint barrier
    that ensures the drone does not collide with the car body.

    We also allow the drone to land if it's sufficiently centered above the car.
    """

    def __init__(
        self,
        xml_file="Drone_ver_1.0/drone-v1.xml",
        frame_skip=5,
        use_safety_filter=True,
        robust_noise_bound=0.05,
        collision_radius=0.35,
        landing_xy_threshold=0.15,
        cbf_relaxation=True
    ):
        """
        :param xml_file: MuJoCo xml model filename
        :param frame_skip: Simulation step skip
        :param use_safety_filter: If True, apply the CBF-based QP to filter actions
        :param robust_noise_bound: Bound on position error for robust safety margin
        :param collision_radius: The effective bounding radius (drone + car) for collision avoidance
        :param landing_xy_threshold: Horizontal threshold to allow final descent
        :param cbf_relaxation: If True, includes a 'delta >= 0' slack in the QP to avoid infeasibility
        """
        utils.EzPickle.__init__(**locals())
        self.use_safety_filter = use_safety_filter
        self.robust_noise_bound = robust_noise_bound
        self.collision_radius = collision_radius
        self.landing_xy_threshold = landing_xy_threshold
        self.cbf_relaxation = cbf_relaxation

        # Turn-off flag used in the original code
        self.turn_off_flag = 0

        # For referencing collisions
        self.car_body_array = ["Landing_box_col"]
        self.drone_body_array = [
            "Main_body_col_1",
            "Main_body_col_2",
            "Main_body_col_3",
            "Main_body_col_4",
        ]
        self.drone_blade_array = [
            "FL_blade_col",
            "FR_blade_col",
            "BL_blade_col",
            "BR_blade_col",
        ]

        # Bookkeeping
        self.dist_between_agents = 0
        self.xy_Distance_between_two_agents = 0

        # Action buffer
        self.action_buffer = np.zeros(4)
        self.action_buffer_2 = np.zeros(4)

        # Provide placeholders for relative heading (unused in this example)
        self.rel_desired_heading_vec = np.array([0, 0, 0])

        # Initialize the MujocoEnv
        mujoco_env.MujocoEnv.__init__(self, xml_file, frame_skip)

    # ---------------------------------------------------------------
    #  Basic environment properties and overrides
    # ---------------------------------------------------------------
    @property
    def done(self):
        return not self.is_healthy

    @property
    def is_healthy(self):
        # Simple checks:
        #  1) Not below some negative threshold
        #  2) Not an extreme pitch
        #  3) Distance constraints
        #  4) No blade collision
        z_ok = (self.state_vector()[2] > -1.9)
        pitch_ok = (abs(self.state_vector()[4]) < 0.8)
        dist_ok = (
            (self.dist_between_agents < 4.0)
            and (self.xy_Distance_between_two_agents < 3.0)
        )

        is_healthy = z_ok and pitch_ok and dist_ok

        # Check for blade collisions
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if geom2 == self.car_body_array[0] and geom1 in self.drone_blade_array:
                print("Blade Collision!! : RESET")
                return False

        return is_healthy

    def viewer_setup(self):
        # Optionally override to position the camera
        if self.viewer is not None:
            self.viewer.cam.lookat[0] = 0
            self.viewer.cam.lookat[1] = 0
            self.viewer.cam.lookat[2] = 1
            self.viewer.cam.distance = 4.0

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)

        self.dist_between_agents = 0
        self.xy_Distance_between_two_agents = 0
        self.action_buffer = np.zeros(4)
        self.action_buffer_2 = np.zeros(4)
        self.turn_off_flag = 0

        return self._get_obs()

    def _get_obs(self):
        # Return the same style observation as older versions:
        # [action_buffer(4), pitch(1), rel_desired_heading(3)]
        pitch = self.state_vector()[4]
        return np.concatenate([self.action_buffer, [pitch], self.rel_desired_heading_vec])

    def step(self, action):
        if self.turn_off_flag == 1:
            # Force gentle descent if we've succeeded
            action = np.array([0, 0, -0.1, 0], dtype=np.float32)
        else:
            action = np.array(action, dtype=np.float32)

        # 1) Filter the action with the robust CBF-based QP
        if self.use_safety_filter:
            safe_action = self.apply_cbf_safety_filter(action)
        else:
            safe_action = action

        self.action_buffer_2 = self.action_buffer.copy()
        self.action_buffer = safe_action.copy()

        # Perform the step in MuJoCo
        self.data.ctrl[:] = safe_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # For consistency with the original environment:
        qpos = np.array(self.data.qpos)
        qvel = np.array(
            [
                safe_action[0],
                safe_action[1],
                safe_action[2],
                0,
                safe_action[3],
                0,
                0,
                0,
                0,
                0,
                0.28,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.set_state(qpos, qvel)

        # Compute reward
        reward = self.compute_reward()

        # done?
        done = self.done

        return self._get_obs(), reward, done, {"total_reward": reward}

    # ---------------------------------------------------------------
    #  Reward function
    # ---------------------------------------------------------------
    def compute_reward(self):
        state = self.state_vector()
        # Drone
        x_drone = state[0] - 1.5
        y_drone = state[1]
        z_drone = state[2] + 2
        # Car
        x_car = state[10] - 0.03
        y_car = state[11]
        z_car = state[12] + 0.1

        drone_pos = np.array([x_drone, y_drone, z_drone])
        car_pos = np.array([x_car, y_car, z_car])

        self.dist_between_agents = np.linalg.norm(drone_pos - car_pos)
        self.xy_Distance_between_two_agents = np.linalg.norm(drone_pos[:2] - car_pos[:2])

        dist_reward = 10 - (6 * self.xy_Distance_between_two_agents + 20 * abs(state[4]))

        land_reward = 0
        if self.xy_Distance_between_two_agents < 0.35:
            # Extra bonus for being close horizontally
            land_reward = 30 / (0.1 + np.abs(drone_pos[2] - car_pos[2]))
            print("Pole!")  # debug message

        # Check if drone main body is fully touching
        touched_set = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if geom2 == self.car_body_array[0] and geom1 in self.drone_body_array:
                touched_set.add(geom1)

        goal_reward = 0
        if len(touched_set) == 4:
            goal_reward = 500000
            self.turn_off_flag = 1
            print("!!!!!!! SUCCESS !!!!!!! : !!!!!!! 4 Points are Touched !!!!!!!")

        # Blade collision penalty
        col_reward = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (geom2 == self.car_body_array[0]) and (geom1 in self.drone_blade_array):
                col_reward = -100
                break

        # Smoothness penalty
        over_reward = -0.2 * np.linalg.norm(self.action_buffer_2 - self.action_buffer)

        total_reward = dist_reward + land_reward + goal_reward + col_reward + over_reward
        return total_reward

    # ---------------------------------------------------------------
    #  CBF-based QP for robust safety
    # ---------------------------------------------------------------
    def apply_cbf_safety_filter(self, raw_action):
        """
        Solve a small QP to find the 'closest' feasible action to raw_action
        that satisfies the robust CBF constraint: h_dot >= -gamma * h,
        for collision avoidance with the car.

        We'll use a minimal continuous-time model for the drone's (x,y,z) and
        interpret (raw_action[0], raw_action[1], raw_action[2]) as acceleration in x,y,z,
        ignoring yaw for the barrier constraint. (yaw is less critical for raw collision).
        """
        dt = self.model.opt.timestep * self.frame_skip

        # Current state
        s = self.state_vector()
        x_drone = s[0] - 1.5
        y_drone = s[1]
        z_drone = s[2] + 2
        vx_drone = s[3]
        vy_drone = s[4]
        vz_drone = s[5]

        # Car state (approx) - from qpos as well
        x_car = s[10] - 0.03
        y_car = s[11]
        z_car = s[12] + 0.1

        # We'll assume the car’s velocity is small or constant. If needed,
        # you can parse qvel or model some known velocity. (Simplified here.)
        vx_car = 0.0
        vy_car = 0.0
        vz_car = 0.0

        # We define the distance-based barrier in squared form for smoothness:
        #   h(x) = (dist^2) - (margin^2)
        # We want h(x) > 0 for safety.
        # For robust approach, we enlarge the margin by robust_noise_bound.
        margin = self.collision_radius + self.robust_noise_bound
        dist_sq = (x_drone - x_car) ** 2 + (y_drone - y_car) ** 2 + (z_drone - z_car) ** 2
        h_val = dist_sq - margin**2

        # If we are within "landing_xy_threshold" horizontally, we relax or skip the constraint
        # to allow the drone to land on the car. This is a design choice (ensures you can land).
        xy_dist = np.linalg.norm([x_drone - x_car, y_drone - y_car])
        if xy_dist < self.landing_xy_threshold:
            # We let the drone proceed to land. Possibly we do not enforce the barrier in z
            # or we keep a smaller margin. We'll skip the constraint if it's truly overhead.
            # Return raw_action to avoid blocking landing.
            return raw_action

        # If h_val is already quite large, we might not need to do anything. But let's do the QP generally.
        # The derivative of dist_sq wrt x, y, z is 2*(x_drone - x_car), etc.
        # We'll treat control in continuous time: a_x, a_y, a_z => next velocity changes.
        # Dot of dist^2 = 2*(p_drone - p_car) dot (v_drone - v_car).
        # Then partial wrt a_x (acc) => 2*(p_drone - p_car) dot ??? + 2*(v_drone - v_car) dot ???

        # For simplicity: h_dot(x, u) = d/dt (dist^2 - margin^2).
        # dist^2 = (p_d - p_c)^T (p_d - p_c).
        # => \dot{dist^2} = 2(p_d - p_c)^\top (v_d - v_c).
        # v_d depends on the action a = [ax, ay, az].
        # We'll do a small Euler approximation for next step. Then add a linear approx in the derivative constraints.

        # We'll define a "gamma" gain for the barrier. If h is negative, we want to push it up quickly.
        gamma = 2.0  # or any positive gain
        # We want h_dot >= - gamma * h_val

        # We define a 3D vector for the position difference p = (x_drone - x_car, y_drone - y_car, z_drone - z_car)
        p_vec = np.array([x_drone - x_car, y_drone - y_car, z_drone - z_car])
        v_vec = np.array([vx_drone - vx_car, vy_drone - vy_car, vz_drone - vz_car])

        # The partial derivative of dist^2 wrt drone accel is 2 * p^T * ??? 
        # Actually, \dot{dist^2} = 2 p^T * v. Then \ddot{dist^2} = 2 [v^T v + p^T a].
        # But let's keep it simpler: we do a direct linear approximation:
        #
        # h_dot ~ 2 p_vec^T v_vec + 2 p_vec^T * a * dt   (where a = [ax, ay, az]).
        # -> The discrete time might be dt. In continuous time we'd see partial derivative w.r.t. a in that expression.
        #
        # We'll define:
        #    h_dot(a) = 2 * p_vec.dot(v_vec) + 2 * p_vec.dot(a) * dt
        #
        # Then our constraint is:
        #    h_dot(a) >= - gamma * h_val
        #
        # i.e. 2 p_vec.dot(v_vec) + 2 p_vec.dot(a)*dt >= - gamma * h_val

        # Let's define the QP:
        #    min  0.5 * ||u - u_des||^2
        #    s.t. 2 p_vec.dot(v_vec) + 2 p_vec.dot(u[0:3])*dt >= - gamma * h_val
        #         (plus optional slack if desired)
        # We do not strongly constrain yaw in the barrier. We only solve for [ax, ay, az].
        # We'll keep yaw = raw_action[3] unmodified. Or we can include it if we want to ensure
        # some yaw-based constraint, but let's keep it simple.

        u_des = raw_action.copy()  # shape (4,)

        # We'll define a QP variable for [ax, ay, az], and keep yaw the same if you wish.
        ax = cp.Variable()
        ay = cp.Variable()
        az = cp.Variable()

        # Slack if desired
        if self.cbf_relaxation:
            delta = cp.Variable(nonneg=True)
        else:
            delta = 0.0  # no slack

        # Build cost:  minimize (1/2)*|| [ax, ay, az] - [u_des[0], u_des[1], u_des[2]] ||^2
        # We'll keep yaw action the same to avoid messing up orientation.
        cost = 0.5 * cp.sum_squares(
            cp.hstack([ax, ay, az]) - u_des[0:3]
        )

        # Build constraint:
        # 2 p_vec.dot(v_vec) + 2 p_vec.dot([ax,ay,az])*dt >= -gamma*h_val - delta
        dot_term = 2 * p_vec.dot(v_vec) + 2 * (p_vec[0] * ax + p_vec[1] * ay + p_vec[2] * az) * dt
        constraint = dot_term >= -gamma * h_val - delta

        # Form the problem
        problem_vars = [ax, ay, az]
        if self.cbf_relaxation:
            problem_vars.append(delta)

        obj = cp.Minimize(cost)
        prob = cp.Problem(obj, [constraint])

        # Solve
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception as e:
            print("CBF QP failed to solve. Using raw action. Error:", e)
            return u_des  # fallback

        # Check status
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # If infeasible or solver error => fallback to raw action
            print("CBF QP infeasible or error: ", prob.status)
            return u_des

        # Extract solution
        ax_sol = ax.value
        ay_sol = ay.value
        az_sol = az.value

        # Reconstruct final safe action
        safe_action = np.array([ax_sol, ay_sol, az_sol, u_des[3]], dtype=np.float32)
        return safe_action
