import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def safety_filter(x, y, theta, u_des, obstacles, v=1.0, w_max=1.0, margin=0.3, lambda_val=1.0):
    """
    Safety filter using Control Barrier Functions (CBF) and a Quadratic Program (QP)
    solved via CVXPY for the Dubins car dynamics.

    Dynamics:
        x_dot = v*cos(theta)
        y_dot = v*sin(theta)
        theta_dot = u
    where u is the steering rate.

    For each obstacle defined by (ox, oy, r), we define a barrier function:
        h(x, y) = (x - ox)^2 + (y - oy)^2 - (r + margin)^2
    Its derivatives (approximated) are:
        h_dot = 2*v*((x-ox)*cos(theta) + (y-oy)*sin(theta))
        LgLfh = 2*v*((y-oy)*cos(theta) - (x-ox)*sin(theta))
        Lf2h  = 2*v**2 + 2*lambda_val*h_dot + lambda_val**2*h
    The CBF condition is:
        Lf2h + LgLfh*u >= 0
    We solve the QP:
        minimize   0.5*(u - u_des)^2
        subject to Lf2h + LgLfh*u >= 0 for each obstacle,
                   u in [-w_max, w_max].
    """
    # Define decision variable
    u = cp.Variable()

    # Define the quadratic cost
    objective = cp.Minimize(0.5 * cp.square(u - u_des))

    # Build constraints for each obstacle
    constraints = []
    for (ox, oy, r) in obstacles:
        h = (x - ox)**2 + (y - oy)**2 - (r + margin)**2
        h_dot = 2 * v * ((x - ox) * np.cos(theta) + (y - oy) * np.sin(theta))
        LgLfh = 2 * v * ((y - oy) * np.cos(theta) - (x - ox) * np.sin(theta))
        Lf2h  = 2 * v**2 + 2 * lambda_val * h_dot + lambda_val**2 * h
        constraints.append(Lf2h + LgLfh * u >= 0)

    # Control input bounds
    constraints.append(u >= -w_max)
    constraints.append(u <= w_max)

    # Solve the QP
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    # If the problem is infeasible, return the nominal control
    if u.value is None:
        return u_des
    return u.value

def dubins_dynamics_step(x, y, theta, u, dt=0.1, v=1.0):
    """
    Propagate the Dubins car dynamics one time step:
        x_next = x + v*cos(theta)*dt
        y_next = y + v*sin(theta)*dt
        theta_next = theta + u*dt
    """
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + u * dt
    return x_next, y_next, theta_next

if __name__ == "__main__":
    # Simulation parameters
    dt = 0.1
    v = 1.0
    w_max = 1.0
    max_iter = 200
    goal_tolerance = 0.1
    lambda_val = 1.0  # CBF gain

    # Define obstacles as (ox, oy, r)
    obstacles = [
        (2.0, 2.0, 0.5),
        (4.0, 1.0, 0.5),
        (3.0, 3.0, 0.5)
    ]
    margin = 0.3  # safety margin around obstacles

    # Start and goal states
    x0, y0, theta0 = 0.0, 0.0, 0.0
    x_goal, y_goal = 5.0, 4.0

    # Nominal controller: simple proportional controller steering towards the goal.
    def nominal_controller(x, y, theta, x_goal, y_goal):
        desired_angle = np.arctan2(y_goal - y, x_goal - x)
        heading_error = desired_angle - theta
        # Wrap error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        u_des = 2.0 * heading_error
        return np.clip(u_des, -w_max, w_max)

    # Histories for simulation: one for safety-filtered trajectory, one for nominal trajectory
    x_hist_safe = [x0]
    y_hist_safe = [y0]
    theta_hist_safe = [theta0]
    u_hist_safe = []

    x_hist_nom = [x0]
    y_hist_nom = [y0]
    theta_hist_nom = [theta0]
    u_hist_nom = []

    # Initialize state variables for both trajectories
    x_safe, y_safe, theta_safe = x0, y0, theta0
    x_nom, y_nom, theta_nom = x0, y0, theta0

    iter_num = 0

    while iter_num < max_iter and np.sqrt((x_safe - x_goal)**2 + (y_safe - y_goal)**2) > goal_tolerance:
        # Compute nominal control for both trajectories based on their own states
        u_nom = nominal_controller(x_nom, y_nom, theta_nom, x_goal, y_goal)
        u_safe_nom = nominal_controller(x_safe, y_safe, theta_safe, x_goal, y_goal)
        
        # Safety filter applied only to the safety trajectory
        u_safe = safety_filter(x_safe, y_safe, theta_safe, u_safe_nom, obstacles,
                               v=v, w_max=w_max, margin=margin, lambda_val=lambda_val)
        
        # Propagate both dynamics
        x_nom, y_nom, theta_nom = dubins_dynamics_step(x_nom, y_nom, theta_nom, u_nom, dt=dt, v=v)
        x_safe, y_safe, theta_safe = dubins_dynamics_step(x_safe, y_safe, theta_safe, u_safe, dt=dt, v=v)
        
        # Store trajectories
        x_hist_nom.append(x_nom)
        y_hist_nom.append(y_nom)
        theta_hist_nom.append(theta_nom)
        u_hist_nom.append(u_nom)
        
        x_hist_safe.append(x_safe)
        y_hist_safe.append(y_safe)
        theta_hist_safe.append(theta_safe)
        u_hist_safe.append(u_safe)
        
        iter_num += 1

    # Plot the trajectories and obstacles
    plt.figure(figsize=(10, 8))
    
    # Plot nominal trajectory
    plt.plot(x_hist_nom, y_hist_nom, 'b--', label='Nominal Trajectory')
    # Plot safety-filtered trajectory
    plt.plot(x_hist_safe, y_hist_safe, 'r-', label='CBF Safety Filtered Trajectory')
    
    # Plot start and goal points
    plt.plot(x0, y0, 'go', label='Start')
    plt.plot(x_goal, y_goal, 'kx', markersize=10, label='Goal')
    
    # Plot obstacles
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r + margin, color='gray', fill=True, alpha=0.3)
        plt.gca().add_patch(circ)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Dubins Car: Nominal vs. CBF Safety Filtered Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
