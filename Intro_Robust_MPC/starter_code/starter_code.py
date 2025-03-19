"""
Starter Code for Dubins Car Control Assignment via Robust MPC and CBF

In this assignment, you will implement:
  1. Standard MPC for the Dubins car.
  2. Robust MPC for the Dubins car.
  3. A Safety Filter using Control Barrier Functions (CBF) solved via CVXPY.

The Dubins car dynamics are:
    xₖ₊₁ = xₖ + v*cos(θₖ)*dt
    yₖ₊₁ = yₖ + v*sin(θₖ)*dt
    θₖ₊₁ = θₖ + uₖ*dt

where uₖ is the steering (control) input.

The environment also contains circular obstacles.
Each obstacle is defined as (ox, oy, r). To ensure safety, a margin is added.

Below you will find function templates with "TODO" comments. Your task is to fill in
the missing parts for each controller method.

Note: This starter code uses CasADi for MPC implementations and CVXPY for the CBF safety filter.
"""

import numpy as np
import casadi as ca
import cvxpy as cp
import matplotlib.pyplot as plt

# ----------------------------
# Dubins Car Dynamics Function
# ----------------------------
def dubins_dynamics(x, y, theta, u, dt, v=1.0):
    """
    Compute the next state of the Dubins car given the current state and control.
    x_next = x + v*cos(theta)*dt
    y_next = y + v*sin(theta)*dt
    theta_next = theta + u*dt
    """
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + u * dt
    return x_next, y_next, theta_next

# ----------------------------
# Standard MPC Implementation
# ----------------------------
def dubins_mpc_step(x0, y0, theta0, x_goal, y_goal, obstacles, 
                    N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3):
    """
    TODO: Implement standard MPC for the Dubins car using CasADi.

    System dynamics:
        xₖ₊₁ = xₖ + v*cos(θₖ)*dt
        yₖ₊₁ = yₖ + v*sin(θₖ)*dt
        θₖ₊₁ = θₖ + uₖ*dt

    Objective:
        Minimize final distance to goal and control effort.

    Constraints:
      - Dynamics (as above)
      - Obstacle avoidance:
            (x - ox)² + (y - oy)² >= (r + margin)²  for each obstacle and time step
      - Control input bounds:
            u ∈ [ -w_max, w_max ]
    
    Use CasADi to formulate and solve the nonlinear programming problem.
    Return the first control input (u₀) of the computed optimal sequence.
    """
    # TODO: Define CasADi symbols for states and controls.
    #       For example:
    #         X = ca.SX.sym('X', N+1)
    #         Y = ca.SX.sym('Y', N+1)
    #         TH = ca.SX.sym('TH', N+1)
    #         U = ca.SX.sym('U', N)
    
    # TODO: Build the objective function (e.g., final distance to goal + control effort).
    
    # TODO: Build the constraints:
    #         - Initial condition constraints.
    #         - Dynamics constraints.
    #         - Obstacle avoidance constraints.
    
    # TODO: Set up the NLP and solve using ca.nlpsol.
    
    # For now, return a dummy control input.
    u_opt = 0.0  # Replace with your computed optimal control input.
    return u_opt

# ----------------------------
# Robust MPC Implementation
# ----------------------------
def robust_dubins_mpc_step(x0, y0, theta0, x_goal, y_goal, obstacles, 
                           N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3, disturbance_bound=0.1):
    """
    TODO: Implement robust MPC for the Dubins car.
    
    Hint: Consider a min-max formulation or tube-based approach where you account for
          worst-case disturbances. You may simulate multiple disturbance scenarios and 
          enforce constraints for all scenarios.
    
    System dynamics and constraints are similar to standard MPC, but must account for
    additive disturbances bounded by disturbance_bound.
    
    Return the first control input (u₀) of the robust control sequence.
    """
    # TODO: Define variables for the nominal trajectory and disturbance scenarios.
    
    # TODO: Build the robust objective and constraints.
    
    # TODO: Solve the robust MPC problem using CasADi.
    
    u_opt = 0.0  # Replace with your robust optimal control input.
    return u_opt

# ----------------------------
# CBF Safety Filter using CVXPY
# ----------------------------
def safety_filter(x, y, theta, u_des, obstacles, v=1.0, w_max=1.0, margin=0.3, lambda_val=1.0):
    """
    TODO: Implement a safety filter using Control Barrier Functions (CBF).
    
    For each obstacle (ox, oy, r), define the barrier function:
        h(x, y) = (x - ox)² + (y - oy)² - (r + margin)²

    Compute its time derivative:
        ḣ = 2*v*((x - ox)*cos(theta) + (y - oy)*sin(theta))

    And an approximation of its second derivative:
        LgLfh = 2*v*((y - oy)*cos(theta) - (x - ox)*sin(theta))
        Lf2h  = 2*v² + 2*lambda_val*ḣ + lambda_val²*h

    The CBF constraint is:
        Lf2h + LgLfh*u >= 0

    Solve the following Quadratic Program (QP) using CVXPY:
        minimize   0.5*(u - u_des)²
        subject to Lf2h + LgLfh*u >= 0, for each obstacle
                   u ∈ [ -w_max, w_max ]
    
    Return the safe control input u_safe.
    """
    # TODO: Set up the CVXPY optimization variable.
    u = cp.Variable()

    # TODO: Define the objective function.
    objective = cp.Minimize(0.5 * cp.square(u - u_des))

    # TODO: Build the constraints for each obstacle.
    constraints = []
    for (ox, oy, r) in obstacles:
        h = (x - ox)**2 + (y - oy)**2 - (r + margin)**2
        h_dot = 2 * v * ((x - ox)*np.cos(theta) + (y - oy)*np.sin(theta))
        LgLfh = 2 * v * ((y - oy)*np.cos(theta) - (x - ox)*np.sin(theta))
        Lf2h  = 2 * v**2 + 2 * lambda_val * h_dot + lambda_val**2 * h
        constraints.append(Lf2h + LgLfh * u >= 0)

    # Add control input bounds.
    constraints += [u >= -w_max, u <= w_max]

    # TODO: Solve the QP.
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    # If the QP fails, use the nominal control.
    if u.value is None:
        return u_des
    return u.value

# ----------------------------
# Nominal Controller (for simulation)
# ----------------------------
def nominal_controller(x, y, theta, x_goal, y_goal, w_max=1.0):
    """
    A simple proportional controller to steer the Dubins car toward the goal.
    Computes a desired steering rate u_des.
    """
    desired_angle = np.arctan2(y_goal - y, x_goal - x)
    heading_error = desired_angle - theta
    # Wrap error to [-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    u_des = 2.0 * heading_error
    return np.clip(u_des, -w_max, w_max)

# ----------------------------
# Simulation Environment
# ----------------------------
if __name__ == "__main__":
    # Simulation parameters
    dt = 0.1          # time step
    v = 1.0           # constant speed
    w_max = 1.0       # maximum steering rate
    max_iter = 200    # maximum simulation iterations
    goal_tolerance = 0.1  # distance threshold for reaching the goal

    # Define obstacles as (ox, oy, r)
    obstacles = [
        (2.0, 2.0, 0.5),
        (4.0, 1.0, 0.5),
        (3.0, 3.0, 0.5)
    ]
    margin = 0.3  # safety margin around obstacles

    # Initial and goal states
    x0, y0, theta0 = 0.0, 0.0, 0.0
    x_goal, y_goal = 5.0, 4.0

    # Initialize trajectories for each control method
    trajectory_mpc = {'x': [x0], 'y': [y0], 'theta': [theta0]}
    trajectory_robust = {'x': [x0], 'y': [y0], 'theta': [theta0]}
    trajectory_cbf = {'x': [x0], 'y': [y0], 'theta': [theta0]}

    # Initialize state variables for each controller
    x_mpc, y_mpc, theta_mpc = x0, y0, theta0
    x_robust, y_robust, theta_robust = x0, y0, theta0
    x_cbf, y_cbf, theta_cbf = x0, y0, theta0

    # Simulation loop
    for i in range(max_iter):
        # Check if goal is reached (use one of the trajectories as reference)
        if np.sqrt((x_mpc - x_goal)**2 + (y_mpc - y_goal)**2) < goal_tolerance:
            break

        # -------- Standard MPC --------
        # TODO: Call your standard MPC function.
        u_mpc = dubins_mpc_step(x_mpc, y_mpc, theta_mpc, x_goal, y_goal, obstacles,
                                N=10, dt=dt, v=v, w_max=w_max, margin=margin)
        x_mpc, y_mpc, theta_mpc = dubins_dynamics(x_mpc, y_mpc, theta_mpc, u_mpc, dt, v)
        trajectory_mpc['x'].append(x_mpc)
        trajectory_mpc['y'].append(y_mpc)
        trajectory_mpc['theta'].append(theta_mpc)

        # -------- Robust MPC --------
        # TODO: Call your robust MPC function.
        u_robust = robust_dubins_mpc_step(x_robust, y_robust, theta_robust, x_goal, y_goal, obstacles,
                                          N=10, dt=dt, v=v, w_max=w_max, margin=margin, disturbance_bound=0.1)
        x_robust, y_robust, theta_robust = dubins_dynamics(x_robust, y_robust, theta_robust, u_robust, dt, v)
        trajectory_robust['x'].append(x_robust)
        trajectory_robust['y'].append(y_robust)
        trajectory_robust['theta'].append(theta_robust)

        # -------- CBF Safety Filter --------
        # Get a nominal control from a simple controller.
        u_nom = nominal_controller(x_cbf, y_cbf, theta_cbf, x_goal, y_goal, w_max)
        # TODO: Apply your CBF safety filter.
        u_cbf = safety_filter(x_cbf, y_cbf, theta_cbf, u_nom, obstacles,
                              v=v, w_max=w_max, margin=margin, lambda_val=1.0)
        x_cbf, y_cbf, theta_cbf = dubins_dynamics(x_cbf, y_cbf, theta_cbf, u_cbf, dt, v)
        trajectory_cbf['x'].append(x_cbf)
        trajectory_cbf['y'].append(y_cbf)
        trajectory_cbf['theta'].append(theta_cbf)

    # ----------------------------
    # Plotting the Results
    # ----------------------------
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory_mpc['x'], trajectory_mpc['y'], 'b-', label='Standard MPC')
    plt.plot(trajectory_robust['x'], trajectory_robust['y'], 'g-', label='Robust MPC')
    plt.plot(trajectory_cbf['x'], trajectory_cbf['y'], 'r-', label='CBF Safety Filter')
    plt.plot(x0, y0, 'ko', label='Start')
    plt.plot(x_goal, y_goal, 'mx', markersize=10, label='Goal')
    # Plot obstacles
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r+margin, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Dubins Car: MPC, Robust MPC, and CBF Safety Filtering')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
