import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

def dubins_mpc_step(x0, y0, th0, x_goal, y_goal, obstacles, N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3):
    """
    Solve a single MPC step for the Dubins car model:
        x_{k+1} = x_k + v*cos(th_k)*dt
        y_{k+1} = y_k + v*sin(th_k)*dt
        th_{k+1} = th_k + w_k*dt
    subject to obstacle avoidance:
        (x_k - x_obs)^2 + (y_k - y_obs)^2 >= (r_obs + margin)^2
    and w_k in [-w_max, w_max].

    Returns:
        w0: the optimal steering rate for the first step.
    """

    # States and controls
    # X: (N+1) points for x
    # Y: (N+1) points for y
    # TH: (N+1) points for theta
    # W: N points for steering rate input
    X = ca.SX.sym('X', N+1)
    Y = ca.SX.sym('Y', N+1)
    TH = ca.SX.sym('TH', N+1)
    W = ca.SX.sym('W', N)

    # Decision variables
    opt_vars = []
    opt_vars.extend([X, Y, TH, W])
    opt_vars = ca.vertcat(*opt_vars)

    # Parameters (initial state, goal)
    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    th_init = ca.SX.sym('th_init')
    x_ref = ca.SX.sym('x_ref')
    y_ref = ca.SX.sym('y_ref')
    param = ca.vertcat(x_init, y_init, th_init, x_ref, y_ref)

    # Constraints and objective
    g = []
    obj = 0

    # Initial conditions
    g.append(X[0] - x_init)
    g.append(Y[0] - y_init)
    g.append(TH[0] - th_init)

    # Build dynamics constraints
    for k in range(N):
        x_next = X[k] + v*ca.cos(TH[k])*dt
        y_next = Y[k] + v*ca.sin(TH[k])*dt
        th_next = TH[k] + W[k]*dt

        g.append(X[k+1] - x_next)
        g.append(Y[k+1] - y_next)
        g.append(TH[k+1] - th_next)

    # Obstacle avoidance constraints
    # For each predicted state, ensure it doesn't collide
    for k in range(N+1):
        for (ox, oy, r) in obstacles:
            g.append(((X[k] - ox)**2 + (Y[k] - oy)**2) - (r + margin)**2)

    # Input (W) constraints
    # We'll enforce w_k in [-w_max, w_max] by setting bounds in the solver
    # Objective: minimize final distance to goal + small penalty on steering
    obj += (X[N] - x_ref)**2 + (Y[N] - y_ref)**2
    for k in range(N):
        obj += 0.01*(W[k])**2

    # Create an NLP solver
    nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': param}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Bounds
    # State variables unbounded, obstacle constraints enforced by g
    lbg = []
    ubg = []
    # initial condition constraints = 0
    for _ in range(3):
        lbg.append(0)
        ubg.append(0)
    # dynamic constraints = 0
    for _ in range(3*N):
        lbg.append(0)
        ubg.append(0)
    # obstacle constraints >= 0
    for _ in range((N+1)*len(obstacles)):
        lbg.append(0)      # (x-ox)^2+(y-oy)^2 - (r+margin)^2 >= 0
        ubg.append(ca.inf)

    # variable bounds
    vars_init = np.zeros(( (N+1)*3 + N ))
    lbx = []
    ubx = []
    # X, Y, TH unbounded in principle
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    # W in [-w_max, w_max]
    for _ in range(N):
        lbx.append(-w_max)
        ubx.append(w_max)

    # Solve
    p_val = np.array([x0, y0, th0, x_goal, y_goal])
    sol = solver(lbx=lbx, ubx=ubx,
                 lbg=lbg, ubg=ubg,
                 p=p_val,
                 x0=vars_init)

    sol_w = sol['x'][(N+1)*3 : (N+1)*3 + N].full().flatten()
    return sol_w[0]

if __name__ == "__main__":
    # Obstacles: list of (x_center, y_center, radius)
    obstacles = [
        (2.0, 2.0, 0.5),
        (4.0, 1.0, 0.5),
        (3.0, 3.0, 0.5)
    ]

    # Simulation params
    dt = 0.1
    N_mpc = 10
    v = 1.0
    w_max = 1.0
    max_iter = 200
    goal_tolerance = 0.1

    # Initial state and goal
    x0, y0, th0 = 0.0, 0.0, 0.0
    x_goal, y_goal = 5.0, 4.0

    x_hist = [x0]
    y_hist = [y0]
    th_hist = [th0]

    x_curr, y_curr, th_curr = x0, y0, th0

    for i in range(max_iter):
        # Check distance to goal
        dist_to_goal = np.sqrt((x_curr - x_goal)**2 + (y_curr - y_goal)**2)
        if dist_to_goal < goal_tolerance:
            break

        # Solve MPC for the next steering rate
        w_opt = dubins_mpc_step(
            x_curr, y_curr, th_curr,
            x_goal, y_goal,
            obstacles,
            N=N_mpc,
            dt=dt,
            v=v,
            w_max=w_max
        )

        # Apply only the first control
        x_next = x_curr + v*np.cos(th_curr)*dt
        y_next = y_curr + v*np.sin(th_curr)*dt
        th_next = th_curr + w_opt*dt

        x_curr, y_curr, th_curr = x_next, y_next, th_next

        x_hist.append(x_curr)
        y_hist.append(y_curr)
        th_hist.append(th_curr)

    # Plot results
    plt.figure(figsize=(6,6))
    plt.plot(x_hist, y_hist, 'b-', label='Trajectory')
    plt.plot(x0, y0, 'go', label='Start')
    plt.plot(x_goal, y_goal, 'rx', label='Goal')

    # Plot obstacles
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r, color='r', fill=True, alpha=0.3)
        plt.gca().add_patch(circ)

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Dubins Car MPC with Obstacle Avoidance')
    plt.show()
