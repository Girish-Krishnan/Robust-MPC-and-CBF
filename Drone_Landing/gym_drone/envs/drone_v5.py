import numpy as np
import mujoco
import cvxpy as cp
from gym import utils
from gym_drone.envs import mujoco_env
import itertools


class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Drone environment that applies a short-horizon robust MPC approach to filter actions.
    Ensures the drone cannot collide with the car, even under bounded noise, 
    while allowing the drone to land on top of the car.
    """

    def __init__(
        self,
        xml_file="Drone_ver_1.0/drone-v1.xml",
        frame_skip=5,
        use_safety_filter=True,
        horizon=3,
        dt=None,
        robust_noise_bound=0.05,
        collision_radius=0.35,
        landing_xy_threshold=0.2,
        max_acc=2.0,
    ):
        """
        :param xml_file: MuJoCo model XML file
        :param frame_skip: Number of physics steps per environment step
        :param use_safety_filter: If True, the robust MPC filter is applied
        :param horizon: Number of discrete steps in the short-horizon MPC
        :param dt: time step for the model. If None, use model.opt.timestep * frame_skip.
        :param robust_noise_bound: Bound on position noise (3D bounding box)
        :param collision_radius: Sum of bounding radii for car + drone
        :param landing_xy_threshold: If horizontally within this threshold, skip or relax collision constraint
        :param max_acc: Maximum absolute acceleration in each axis (x,y,z)
        """
        utils.EzPickle.__init__(**locals())

        self.use_safety_filter = use_safety_filter
        self.robust_noise_bound = robust_noise_bound
        self.collision_radius = collision_radius
        self.landing_xy_threshold = landing_xy_threshold
        self.horizon = horizon
        self.max_acc = max_acc

        # "Turn off" the drone after success
        self.turn_off_flag = 0

        # Distances for logging
        self.dist_between_agents = 0
        self.xy_Distance_between_two_agents = 0

        # Action buffers
        self.action_buffer = np.zeros(4)
        self.action_buffer_2 = np.zeros(4)

        # Just to be consistent with prior versions
        self.rel_desired_heading_vec = np.array([0, 0, 0])

        # Car vs. drone body for collision checks
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

        super().__init__(xml_file, frame_skip)

        if dt is None:
            self._my_dt = self.model.opt.timestep * self.frame_skip
        else:
            self._my_dt = dt

    # ------------------------------------------------------------------
    # Environment overrides
    # ------------------------------------------------------------------
    
    @property
    def done(self):
        return not self.is_healthy

    @property
    def is_healthy(self):
        # Basic checks
        z_ok = (self.state_vector()[2] > -1.9)
        pitch_ok = (abs(self.state_vector()[4]) < 0.8)
        dist_ok = (
            (self.dist_between_agents < 4.0)
            and (self.xy_Distance_between_two_agents < 3.0)
        )

        is_healthy = z_ok and pitch_ok and dist_ok

        # Blade collision
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if geom2 in self.car_body_array and geom1 in self.drone_blade_array:
                print("Blade Collision!! : RESET")
                return False

        return is_healthy

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)

        self.dist_between_agents = 0
        self.xy_Distance_between_two_agents = 0
        self.action_buffer = np.zeros(4)
        self.action_buffer_2 = np.zeros(4)
        self.turn_off_flag = 0

        return self._get_obs()

    def _get_obs(self):
        # Return: action_buffer(4), pitch(1), rel_desired_heading(3)
        pitch = self.state_vector()[4]
        return np.concatenate([self.action_buffer, [pitch], self.rel_desired_heading_vec])

    def step(self, action):
        if self.turn_off_flag == 1:
            # If success locked, override action to gentle descent
            action = np.array([0, 0, -0.1, 0], dtype=np.float32)
        else:
            action = np.array(action, dtype=np.float32)

        # Apply robust MPC if enabled
        if self.use_safety_filter:
            safe_action = self.apply_robust_mpc_filter(action)
        else:
            safe_action = action

        self.action_buffer_2 = self.action_buffer.copy()
        self.action_buffer = safe_action.copy()

        # Step simulation
        self.data.ctrl[:] = safe_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # For consistency with older code base:
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

        reward = self.compute_reward()
        done = self.done
        info = {"total_reward": reward}
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------
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
            land_reward = 30 / (0.1 + abs(z_drone - z_car))
            print("Pole!")  # debug message

        # Check if main body fully touches
        touched_set = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (geom2 in self.car_body_array) and (geom1 in self.drone_body_array):
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
            if (geom2 in self.car_body_array) and (geom1 in self.drone_blade_array):
                col_reward = -100
                break

        # Smoothness penalty
        over_reward = -0.2 * np.linalg.norm(self.action_buffer_2 - self.action_buffer)

        return dist_reward + land_reward + goal_reward + col_reward + over_reward

    # ------------------------------------------------------------------
    # The robust MPC filter
    # ------------------------------------------------------------------
    def apply_robust_mpc_filter(self, raw_action):
        """
        Solve a short-horizon robust MPC to find the best first-step control that ensures
        no collision over the horizon, under bounded noise. We do a scenario-based approach 
        and keep states as CVXPY Variables, so constraints are symbolic (no Python booleans!).
        """
        # If horizontally close enough, allow landing (skip robust constraints).
        if self._allow_landing():
            return raw_action

        # Solve short-horizon robust MPC
        a0, status = self._solve_short_horizon_mpc(raw_action)
        if status not in ["optimal", "optimal_inaccurate"]:
            # fallback if infeasible or solver error
            return raw_action

        # Return first step acceleration with yaw from raw_action
        safe_action = np.array([a0[0], a0[1], a0[2], raw_action[3]], dtype=np.float32)
        return safe_action

    def _allow_landing(self):
        """
        If the drone is horizontally within 'landing_xy_threshold' of the car,
        skip the collision constraints so it can actually land.
        """
        s = self.state_vector()
        x_drone = s[0] - 1.5
        y_drone = s[1]
        x_car = s[10] - 0.03
        y_car = s[11]
        xy_dist = np.linalg.norm([x_drone - x_car, y_drone - y_car])
        return (xy_dist < self.landing_xy_threshold)

    def _solve_short_horizon_mpc(self, raw_action):
        """
        Uses scenario-based robust MPC with horizon self.horizon steps.
        We define CVXPY Variables for (x[k], y[k], z[k], vx[k], vy[k], vz[k]) 
        for k=0..N, plus a[k] in R^3 for k=0..N-1.

        We penalize ||a[0] - raw_action[:3]||^2 so the final solution stays close 
        to the policy's intended action, and then pick the solution's a0.

        Return: (a0_solution, solver_status).
        """
        N = self.horizon
        dt = self._my_dt if hasattr(self, "_my_dt") else self.model.opt.timestep * self.frame_skip
        noise_b = self.robust_noise_bound
        margin = self.collision_radius

        # =========== GET CURRENT STATE ===========
        s = self.state_vector()
        # Drone pos
        x0 = s[0] - 1.5
        y0 = s[1]
        z0 = s[2] + 2
        # Drone vel
        vx0 = s[3]
        vy0 = s[4]
        vz0 = s[5]
        # Car pos (assuming low or zero velocity)
        x_car = s[10] - 0.03
        y_car = s[11]
        z_car = s[12] + 0.1
        vx_car = 0.0
        vy_car = 0.0
        vz_car = 0.0

        # =========== CVXPY DECISION VARIABLES ===========
        # For each k in 0..N, states: x[k], y[k], z[k], vx[k], vy[k], vz[k]
        x = cp.Variable(N + 1)
        y = cp.Variable(N + 1)
        z = cp.Variable(N + 1)
        vx = cp.Variable(N + 1)
        vy = cp.Variable(N + 1)
        vz = cp.Variable(N + 1)

        # For each k in 0..N-1, acceleration a[k] in R^3
        a = [cp.Variable(3) for _ in range(N)]

        constraints = []

        # =========== INITIAL CONDITIONS ===========
        constraints += [
            x[0] == x0,
            y[0] == y0,
            z[0] == z0,
            vx[0] == vx0,
            vy[0] == vy0,
            vz[0] == vz0,
        ]

        # =========== DYNAMICS CONSTRAINTS ===========
        # x_{k+1} = x_k + vx_k * dt + 0.5 * a[k][0] * dt^2
        # vx_{k+1} = vx_k + a[k][0] * dt
        # similarly for y,z
        for k in range(N):
            constraints += [
                x[k + 1] == x[k] + vx[k] * dt + 0.5 * a[k][0] * (dt**2),
                y[k + 1] == y[k] + vy[k] * dt + 0.5 * a[k][1] * (dt**2),
                z[k + 1] == z[k] + vz[k] * dt + 0.5 * a[k][2] * (dt**2),

                vx[k + 1] == vx[k] + a[k][0] * dt,
                vy[k + 1] == vy[k] + a[k][1] * dt,
                vz[k + 1] == vz[k] + a[k][2] * dt,
            ]

            # Acceleration limits
            constraints += [
                a[k][0] <= self.max_acc,
                a[k][0] >= -self.max_acc,
                a[k][1] <= self.max_acc,
                a[k][1] >= -self.max_acc,
                a[k][2] <= self.max_acc,
                a[k][2] >= -self.max_acc,
            ]

        # =========== ROBUST COLLISION CONSTRAINTS ===========
        # scenario-based approach: For each time k=0..N, for each corner of noise
        corners = list(itertools.product([-1, 1], repeat=3))
        for k in range(N + 1):
            for corner in corners:
                nx = corner[0] * noise_b
                ny = corner[1] * noise_b
                nz = corner[2] * noise_b

                # Car position for scenario
                car_x_scen = x_car + vx_car * (k * dt) + nx
                car_y_scen = y_car + vy_car * (k * dt) + ny
                car_z_scen = z_car + vz_car * (k * dt) + nz

                # Enforce dist^2 >= margin^2 => ( x[k] - car_x_scen )^2 + ... >= margin^2
                dist_sq_expr = cp.square(x[k] - car_x_scen) + \
                               cp.square(y[k] - car_y_scen) + \
                               cp.square(z[k] - car_z_scen)
                constraints += [dist_sq_expr >= margin**2]

        # =========== COST FUNCTION ===========
        # We want to stay close to raw_action in the first step (a0),
        # plus maybe small penalties on subsequent steps so they don't blow up.
        cost_expr = 0.5 * cp.sum_squares(a[0] - raw_action[:3])
        for k in range(1, N):
            cost_expr += 0.01 * cp.sum_squares(a[k])  # small reg

        objective = cp.Minimize(cost_expr)

        # =========== SOLVE PROBLEM ===========
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception as e:
            print("Robust MPC solver failed:", e)
            return (raw_action[:3], "error")

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print("Robust MPC infeasible or error:", prob.status)
            return (raw_action[:3], prob.status)

        # Extract first-step acceleration
        a0_sol = a[0].value  # shape (3,)
        return (a0_sol, prob.status)
