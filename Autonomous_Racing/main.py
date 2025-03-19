import os
import argparse
import yaml
import torch
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env import CarEnv
from dm_control import viewer, composer
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import MultiInputPolicy
import torch as th
import torch.nn as nn
import cv2
from torch_geometric.nn import knn_graph
# from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch_geometric.nn import GraphConv, EdgeConv, knn_graph
from torch_geometric.utils import add_self_loops
from evaluation import Evaluator
import re
import matplotlib.pyplot as plt
import open3d as o3d
import casadi as ca
import time
last_qp_time = 0

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

import cvxpy as cp

class RobustMPC:
    def __init__(self, horizon=5, dt=0.1):
        self.horizon = horizon  # Number of steps in the future to optimize
        self.dt = dt  # Time step

        # Car dynamics model (same as RL)
        self.L = 0.2965  # Wheelbase

        # MPC constraints
        self.max_steering = np.radians(30)  # ±30 degrees
        self.max_throttle = 1.0  # Max acceleration
        self.min_throttle = -1.0  # Max braking

        # Disturbance bounds (for robust MPC)
        self.disturbance_bound = 0.05  # Assume up to ±5% uncertainty

    def predict_dynamics(self, X, U):
        # X and U are CasADi MX variables
        x = X  # [x, y, theta]^T
        u = U  # [steering, throttle]^T

        dx = 0.1 * (u[1] * ca.cos(x[2]))
        dy = 0.1 * (u[1] * ca.sin(x[2]))
        dtheta = 0.1 * ((u[1] * ca.tan(u[0])) / 0.2965)

        # Return next state as a CasADi expression
        x_next = x + ca.vertcat(dx, dy, dtheta)
        return x_next

    def optimize_control(self, x0, u_ref):
        """ Solve the MPC optimization problem """

        opti = ca.Opti()  # Create an optimization environment

        # Define optimization variables
        U = opti.variable(2, self.horizon)  # Control sequence (steering, throttle)
        X = opti.variable(3, self.horizon+1)  # State trajectory (x, y, theta)

        # Initial state constraint
        opti.subject_to(X[:, 0] == x0)

        # Cost function
        cost = 0

        for t in range(self.horizon):
            # Apply kinematic model dynamics
            x_next = self.predict_dynamics(X[:, t], U[:, t])

            # Enforce state transitions
            opti.subject_to(X[:, t+1] == x_next)

            # Control constraints
            opti.subject_to(U[0, t] <= self.max_steering)
            opti.subject_to(U[0, t] >= -self.max_steering)
            opti.subject_to(U[1, t] <= self.max_throttle)
            opti.subject_to(U[1, t] >= self.min_throttle)

            # Cost function: minimize deviation from RL+CBF action
            cost += ca.sumsqr(U[:, t] - u_ref)

            # Smoothness: Penalize large control changes
            if t > 0:
                cost += 10 * ca.sumsqr(U[:, t] - U[:, t-1])

        # Define optimization objective
        opti.minimize(cost)

        # Solver settings
        opti.solver('ipopt')

        # Set initial conditions
        opti.set_initial(X[:, 0], x0)

        try:
            sol = opti.solve()
            u_opt = sol.value(U[:, 0])  # Apply only the first action
        except:
            print("MPC solver failed, falling back to RL+CBF action")
            u_opt = u_ref  # Default to RL+CBF action if solver fails

        return u_opt

def policy(timestep, model, model_inputs):
    global last_qp_time
    vec_obs = []
    depth_obs = None
    point_cloud_obs = None

    # Collect vector observations
    if "pose" in model_inputs:
        pose_obs = timestep.observation['car/body_pose_2d']
        vec_obs += list(pose_obs[:3])
    if "velocity" in model_inputs:
        velocity = timestep.observation['car/body_vel_2d']
        vec_obs += [np.linalg.norm(velocity)]
    if "steering" in model_inputs:
        vec_obs += [timestep.observation['car/steering_pos'][0]]

    # Collect depth observation
    if "depth" in model_inputs:
        depth_obs = timestep.observation['car/realsense_camera'].astype(np.float32)
        cv2.imshow("Depth Map", cv2.convertScaleAbs(depth_obs, alpha=0.15))
        cv2.waitKey(1)

    # Collect point cloud observation
    if "point_cloud" in model_inputs:
        point_cloud_obs = timestep.observation['car/compute_point_cloud']

    # Construct observation dictionary
    observation = {}
    if len(vec_obs) > 0:
        observation["vec"] = np.array(vec_obs)
    if depth_obs is not None:
        observation["depth"] = depth_obs
    if point_cloud_obs is not None:
        observation["point_cloud"] = point_cloud_obs

    # Get point cloud for obstacle mapping
    point_cloud_obs = timestep.observation['car/compute_point_cloud']
    vec_obs = timestep.observation['car/body_pose_2d']
    obstacle_map = None
    if point_cloud_obs is not None:
        # Use open3d to visualize point cloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud_obs)
        # o3d.visualization.draw_geometries([pcd])

        max_y_val = np.max(point_cloud_obs[:, 1])
        point_cloud_obs = point_cloud_obs[point_cloud_obs[:, 1] < max_y_val - 0.02]
        point_cloud_obs = point_cloud_obs[np.linalg.norm(point_cloud_obs, axis=1) > 0.2]

        obstacle_map = point_cloud_obs[:, 0], point_cloud_obs[:, 2]  # (x, z) in car frame

        # Visualize
        # plt.clf()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        # plt.scatter(point_cloud_obs[:, 0], point_cloud_obs[:, 2], s=1, c='b')
        # plt.scatter(0, 0, c='r')  # Car at origin
        # plt.show(block=False)
        # plt.pause(0.01)

    # Get RL action
    action, _ = model.predict(observation, deterministic=True)
    steering_rl, throttle_rl = action

    return np.array([steering_rl, throttle_rl])

    # ---------- CBF Safety Filtering ----------
    if obstacle_map is not None and (time.time() - last_qp_time) > 0.5:
        last_qp_time = time.time()
        # Convert obstacle map to NumPy arrays
        obs_x, obs_z = obstacle_map
        obstacles = np.column_stack((obs_x, obs_z))

        # Define car safety parameters
        safe_margin = 0.5 + 0.5 * np.abs(throttle_rl)  # Minimum allowed distance to obstacles (radius + buffer)
        
        # Define CBF safety constraints
        safety_constraints = []
        for obs in obstacles:
            obs_x, obs_z = obs
            distance = np.sqrt(obs_x**2 + obs_z**2)  # Distance from car (origin)

            # Control Barrier Function (CBF) condition: h(x) = distance - safe_margin > 0
            h_x = distance - safe_margin
            if h_x < 0:  # If unsafe, add a constraint
                safety_constraints.append((obs_x / distance, obs_z / distance, h_x))

        # If there are safety constraints, solve QP
        if safety_constraints:
            # Define optimization variables (adjusted steering, throttle)
            delta_steering = cp.Variable()
            delta_throttle = cp.Variable()

            # Objective: Minimize deviation from RL action
            objective = cp.Minimize(
                (delta_steering - steering_rl)**2 + (delta_throttle - throttle_rl)**2
                # + 1e-4 * (delta_steering**2 + delta_throttle**2)
                )
            
            alpha = 2.0 # gain term for safety margin

            # Constraints: Ensure h(x) remains positive
            constraints = []
            for nx, nz, h_x in safety_constraints:
                constraints.append(
                    # nx * delta_steering + nz * delta_throttle + h_x >= 0
                    nx * (delta_throttle + np.cos(vec_obs[2])) 
                    + nz * (delta_throttle + np.sin(vec_obs[2])) + alpha * h_x >= 0
                )

            # print(f"Num safety constraints: {len(safety_constraints)}")

            # Solve QP
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve()
                steering_safe = delta_steering.value
                throttle_safe = delta_throttle.value
            except:
                print("QP solver failed, falling back to RL action")
                steering_safe, throttle_safe = steering_rl, throttle_rl
        else:
            steering_safe, throttle_safe = steering_rl, throttle_rl
    else:
        steering_safe, throttle_safe = steering_rl, throttle_rl

    # if RL != safe action, print
    if steering_rl != steering_safe or throttle_rl != throttle_safe:
        print(f"Unsafe action: RL: ({steering_rl}, {throttle_rl}), Safe: ({steering_safe}, {throttle_safe})")

    # Run Robust MPC optimization
    # mpc_controller = RobustMPC(horizon=5, dt=0.1)
    # x0 = np.array(vec_obs[:3])  # Current car state
    # u_ref = np.array([steering_safe, throttle_safe])  # Safe action from CBF
    # u_mpc = mpc_controller.optimize_control(x0, u_ref)

    # return u_mpc

    return np.array([steering_safe, throttle_safe])


class ActionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionLoggerCallback, self).__init__(verbose)
        # self.prev_action = None

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        observations = self.locals["new_obs"]
        rewards = self.locals["rewards"]
        
        print(f"Step: {self.num_timesteps}")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print("------")

        # if self.prev_action is None:
        #     self.prev_action = actions

        # # Compute difference with the previous action
        # action_diff = np.linalg.norm(actions - self.prev_action)
        # # Add the penalty to the reward
        # self.locals["rewards"] -= 10 * action_diff
        # print(f"Action Difference Penalty: {10 * action_diff}")

        # # Update previous action
        # self.prev_action = self.locals["actions"].copy()
        
        return True

class GraphFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, k=8):
        super(GraphFeatureExtractor, self).__init__()
        self.k = k

        # Graph Convolutions
        self.conv1 = GraphConv(input_dim, hidden_dims[0])
        self.conv2 = GraphConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GraphConv(hidden_dims[1], output_dim)

    def forward(self, point_cloud):
        batch_size, num_points, feature_dim = point_cloud.shape
        device = point_cloud.device

        # Flatten batch for KNN graph
        point_cloud_flat = point_cloud.view(-1, feature_dim)
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)

        # Compute KNN graph
        edge_index = knn_graph(point_cloud_flat, k=self.k, batch=batch_indices, loop=True)

        # Apply Graph Conv layers
        x = self.conv1(point_cloud_flat, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        # Reshape back to batch format
        return x.view(batch_size, num_points, -1)
    
class EdgeConvFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(EdgeConvFeatureExtractor, self).__init__()

        self.edge_conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        ), aggr="max")

        self.edge_conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        ), aggr="max")

    def forward(self, point_cloud):
        batch_size, num_points, feature_dim = point_cloud.shape
        point_cloud_flat = point_cloud.view(-1, feature_dim)

        # KNN Graph
        edge_index = knn_graph(point_cloud_flat, k=8, loop=True)

        # Apply EdgeConv
        x = self.edge_conv1(point_cloud_flat, edge_index).relu()
        x = self.edge_conv2(x, edge_index)

        return x.view(batch_size, num_points, -1)
    
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)

        self.outputs = {}  # Dictionary to store layer outputs for debugging

        # Process vector inputs
        self.vec_network = None
        if 'vec' in observation_space.keys():
            self.vec_network = nn.Sequential(
                nn.Linear(observation_space['vec'].shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.vec_network[-1].register_forward_hook(self._hook("vec_features"))

        # Process depth inputs
        self.cnn = None
        depth_dim = 0
        if 'depth' in observation_space.keys():
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                depth_sample = observation_space['depth'].sample()[None, :, :, 0]
                depth_sample = torch.tensor(depth_sample).float()
                depth_sample = self.normalize_depth_map(depth_sample).unsqueeze(1)
                depth_dim = self.cnn(depth_sample).shape[1]

        # Process point cloud inputs
        self.point_cloud_extractor = None
        point_cloud_dim = 0
        if 'point_cloud' in observation_space.keys():
            self.point_cloud_extractor = GraphFeatureExtractor(
                input_dim=3, 
                hidden_dims=[64, 64], 
                output_dim=256, 
                k=8
            )

            point_cloud_dim = 256

        # Determine the combined input size
        vec_dim = 64 if 'vec' in observation_space.keys() else 0
        combined_input_dim = vec_dim + depth_dim + point_cloud_dim

        # Combined network
        self.combined_network = nn.Sequential(
            nn.Linear(combined_input_dim, features_dim),
            nn.ReLU()
        )
        self.combined_network[-1].register_forward_hook(self._hook("combined_features"))

    def _hook(self, layer_name):
        """
        Hook to store outputs for debugging purposes.
        """
        def hook(module, input, output):
            self.outputs[layer_name] = output.detach().cpu().numpy()
        return hook

    def normalize_depth_map(self, depth_map):
        """
        Normalize depth map values to the range [0, 1].
        """
        min_val = depth_map.min()
        max_val = depth_map.max()
        return (depth_map - min_val) / (max_val - min_val)

    def forward(self, observations):
        """
        Forward pass to process observations and extract features.
        """
        # Process vector input
        vec_features = None
        if 'vec' in observations.keys():
            vec_features = self.vec_network(observations['vec'])

        # Process depth input
        depth_features = None
        if 'depth' in observations.keys():
            depth_map = observations['depth'][:, :, :, 0]
            depth_map = self.normalize_depth_map(depth_map).unsqueeze(1)
            depth_features = self.cnn(depth_map)

        # Process point cloud input
        point_cloud_features = None
        if 'point_cloud' in observations.keys():
            point_cloud = observations['point_cloud']  # Shape: [num_envs, N, 3]
            point_cloud_features = self.point_cloud_extractor(point_cloud)  # GNN-based extraction
            point_cloud_features = torch.mean(point_cloud_features, dim=1)  # Global feature aggregation

        # Concatenate features
        features = [f for f in [vec_features, depth_features, point_cloud_features] if f is not None]
        combined_features = torch.cat(features, dim=1)

        # Pass through the combined network
        final_output = self.combined_network(combined_features)
        self.outputs["final_output"] = final_output.detach().cpu().numpy()
        return final_output

def evaluate_model(model, vec_input, depth_input):
    observation = {"vec": vec_input, "depth": depth_input}
    model.predict(observation, deterministic=True)
    return model.policy.actor.features_extractor.outputs

def make_env(num_obstacles, rank, log_dir=None, goal_position=None, scenario="no-goal", model_inputs=CarEnv.ALL_MODEL_INPUTS):
    """
    Utility function to create a single instance of the CarEnv environment.
    :param num_obstacles: (int) Number of obstacles in the environment
    :param rank: (int) Rank of the environment (used for seeding)
    :param log_dir: (str) Directory to save logs
    :param goal_position: (list) Goal position for the environment
    :param scenario: (str) Scenario to use for the environment
    :param model_inputs: (list) List of model inputs to use
    """
    def _init():
        env = CarEnv(num_obstacles=num_obstacles, goal_position=goal_position, scenario=scenario, model_inputs=model_inputs)
        env.seed(rank)  # Seed the environment for reproducibility
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init

def ensure_dir_exists(dir_path):
    """
    Ensure that the directory exists, create if it doesn't.
    :param dir_path: (str) Path of the directory to check/create
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a model.")
    parser.add_argument('--model_type', type=str, choices=["PPO", "SAC", "RecurrentPPO"], help='Type of model to use', default="SAC")
    parser.add_argument('--scenario', type=str, choices=["goal", "no-goal"], help='Choose navigation scenario', default="no-goal")
    parser.add_argument('--goal_position', type=str, help='Comma-separated goal coordinates for goal-based navigation (e.g., "5,2")', default=None)
    parser.add_argument('--config_path', type=str, help='Path to the YAML config file', default="config/config_sac.yaml")

    parser.add_argument('--model_path', type=str, help='Path to the saved model (.zip) to continue training or for evaluation', default=None)
    parser.add_argument('--log_dir', type=str, help='Directory to save logs and models', default='./my_experiment/')
    parser.add_argument('--file_name', type=str, help='Base name for saved model and logs', default='model')
    parser.add_argument('--simulate', action='store_true', help='Run in evaluation mode')

    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--seed', type=int, help='The random seed used to generate environment', default=0)
    parser.add_argument('--num_eval_envs', type=int, help='Number of environments to run evaluations in', default=10)
    parser.add_argument('--num_eval_episodes', type=int, help='Number of episodes to run evaluations for in each environment', default=1)

    args = parser.parse_args()

    # Set up directories
    tensorboard_log_dir = os.path.join(args.log_dir, "tensorboard", args.file_name)
    models_dir = os.path.join(args.log_dir, "models")
    logs_dir = os.path.join(args.log_dir, "logs")

    ensure_dir_exists(tensorboard_log_dir)
    ensure_dir_exists(models_dir)
    ensure_dir_exists(logs_dir)

    # Load YAML config
    config = load_yaml(args.config_path)
    training_params = config.get("training", {})
    model_params = config.get("model", {})

    goal_position = None
    if args.goal_position:
        goal_position = np.array([float(x) for x in args.goal_position.split(",")])

    model_class = {"PPO": PPO, "SAC": SAC, "RecurrentPPO": RecurrentPPO}[args.model_type]

    if args.eval:
        # Evaluation Mode
        model = model_class.load(args.model_path)
        try:
            summary_prefix = re.findall(r"\/([^\/]+).zip", args.model_path)[0]
        except:
            summary_prefix = "dummy"
        
        evaluator = Evaluator(None, logs_dir, episodes=args.num_eval_episodes, prefix=summary_prefix, time_limit=1000)
        seedgen = np.random.SeedSequence(entropy=args.seed)
        seeds = seedgen.generate_state(args.num_eval_envs)
        for seed in seeds:
            env = CarEnv(num_obstacles=training_params["num_obstacles"], goal_position=goal_position, 
                          scenario=args.scenario, 
                         model_inputs=training_params["model_inputs"], random_seed=int(seed))
            original_env = env.original_env
            evaluator.set_environment(original_env, seed)
            print(f'Start evaluating in environment with seed {seed}')
            timestep = original_env.reset()
            for _ in range(1000):
                action = policy(timestep, model, model_inputs=training_params["model_inputs"], evaluator = evaluator)
                timestep = original_env.step(action)
                if evaluator.check_finish():
                    break
            
        evaluator.export_csv()

    elif args.simulate:
        # Simulation Mode
        model = model_class.load(args.model_path)
        
        env = CarEnv(num_obstacles=training_params["num_obstacles"], goal_position=goal_position, scenario=args.scenario, model_inputs=training_params["model_inputs"])
        task = env.task
        original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

        viewer.launch(original_env, policy=lambda timestep: policy(timestep, model, model_inputs=training_params["model_inputs"]))

    else:
        # Training Mode
        env = DummyVecEnv([make_env(
                    num_obstacles=training_params["num_obstacles"], 
                    rank=i, 
                    log_dir=logs_dir, 
                    goal_position=goal_position, 
                    scenario=args.scenario, 
                    model_inputs=training_params["model_inputs"]) 
                        for i in range(training_params["num_envs"])])
        
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        device = torch.device(training_params["device"])

        if args.model_path:
            print(f"Loading model from {args.model_path}")
            model = model_class.load(args.model_path, env=env, device=device)
        else:
            print("Training a new model")

            model = model_class(
                    env=env,
                    policy_kwargs=dict(
                        features_extractor_class=FeatureExtractor,
                        features_extractor_kwargs=dict(features_dim=128),
                        net_arch=[256, 256], # 2 layers of 256 units for the latent policy network
                    ),
                    tensorboard_log=tensorboard_log_dir,
                    device=device,
                    **model_params
                )

        callback = ActionLoggerCallback(verbose=1)

        model.learn(total_timesteps=training_params["total_timesteps"], callback=callback, progress_bar=True)

        model_save_path = os.path.join(models_dir, f"{args.file_name}.zip")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        env.save(os.path.join(logs_dir, f"{args.file_name}_vecnormalize.pkl"))

if __name__ == "__main__":
    main()
