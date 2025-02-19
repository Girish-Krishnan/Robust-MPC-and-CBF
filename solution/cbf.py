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

    # Histories for simulation
    x_hist = [x0]
    y_hist = [y0]
    theta_hist = [theta0]
    u_hist = []

    x_curr, y_curr, theta_curr = x0, y0, theta0
    iter_num = 0

    while iter_num < max_iter and np.sqrt((x_curr - x_goal)**2 + (y_curr - y_goal)**2) > goal_tolerance:
        u_nom = nominal_controller(x_curr, y_curr, theta_curr, x_goal, y_goal)
        # Apply safety filter via CBF-QP using CVXPY
        u_safe = safety_filter(x_curr, y_curr, theta_curr, u_nom, obstacles,
                               v=v, w_max=w_max, margin=margin, lambda_val=lambda_val)
        # Propagate the dynamics
        x_next, y_next, theta_next = dubins_dynamics_step(x_curr, y_curr, theta_curr, u_safe, dt=dt, v=v)
        x_hist.append(x_next)
        y_hist.append(y_next)
        theta_hist.append(theta_next)
        u_hist.append(u_safe)
        x_curr, y_curr, theta_curr = x_next, y_next, theta_next
        iter_num += 1

    # Plot the trajectory and obstacles
    plt.figure(figsize=(8, 8))
    plt.plot(x_hist, y_hist, 'b-', label='Trajectory with Safety Filter')
    plt.plot(x0, y0, 'go', label='Start')
    plt.plot(x_goal, y_goal, 'rx', markersize=10, label='Goal')
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r + margin, color='r', fill=True, alpha=0.3)
        plt.gca().add_patch(circ)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Dubins Car with Safety Filter using CBF and CVXPY QP')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
