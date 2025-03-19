import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

def dubins_mpc_step(x0, y0, th0, x_goal, y_goal, obstacles, N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3):
    """
    Standard MPC for the Dubins car.
    """
    X = ca.SX.sym('X', N+1)
    Y = ca.SX.sym('Y', N+1)
    TH = ca.SX.sym('TH', N+1)
    W = ca.SX.sym('W', N)

    opt_vars = ca.vertcat(X, Y, TH, W)

    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    th_init = ca.SX.sym('th_init')
    x_ref = ca.SX.sym('x_ref')
    y_ref = ca.SX.sym('y_ref')
    param = ca.vertcat(x_init, y_init, th_init, x_ref, y_ref)

    g = []
    obj = 0

    # Initial condition
    g.append(X[0] - x_init)
    g.append(Y[0] - y_init)
    g.append(TH[0] - th_init)

    # Dynamics constraints
    for k in range(N):
        x_next = X[k] + v * ca.cos(TH[k]) * dt
        y_next = Y[k] + v * ca.sin(TH[k]) * dt
        th_next = TH[k] + W[k] * dt
        g.append(X[k+1] - x_next)
        g.append(Y[k+1] - y_next)
        g.append(TH[k+1] - th_next)

    # Obstacle constraints (no tightening)
    for k in range(N+1):
        for (ox, oy, r) in obstacles:
            g.append(((X[k] - ox)**2 + (Y[k] - oy)**2) - (r + margin)**2)

    # Objective: final distance + small control penalty
    obj += (X[N] - x_ref)**2 + (Y[N] - y_ref)**2
    for k in range(N):
        obj += 0.01 * (W[k])**2

    nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': param}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    lbg = []
    ubg = []
    # Initial condition constraints
    for _ in range(3):
        lbg.append(0)
        ubg.append(0)
    # Dynamics constraints (3 per step)
    for _ in range(3 * N):
        lbg.append(0)
        ubg.append(0)
    # Obstacle constraints: must be >= 0
    for _ in range((N+1) * len(obstacles)):
        lbg.append(0)
        ubg.append(ca.inf)

    vars_init = np.zeros(((N+1)*3 + N,))
    lbx = []
    ubx = []
    # X, Y, TH: unbounded
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    # W: bounded
    for _ in range(N):
        lbx.append(-w_max)
        ubx.append(w_max)

    p_val = np.array([x0, y0, th0, x_goal, y_goal])
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val, x0=vars_init)
    sol_w = sol['x'][(N+1)*3:(N+1)*3+N].full().flatten()
    return sol_w[0]

def tube_based_robust_mpc_step(x0, y0, th0, x_goal, y_goal, obstacles, N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3, d_max=0.1, tube_margin=0.1):
    """
    Tube-based robust MPC.
    This method solves a nominal MPC problem with tightened obstacle constraints
    to account for bounded additive disturbances.
    The tightened obstacles have an effective radius increased by 'tube_margin'.
    """
    X = ca.SX.sym('X', N+1)
    Y = ca.SX.sym('Y', N+1)
    TH = ca.SX.sym('TH', N+1)
    W = ca.SX.sym('W', N)

    opt_vars = ca.vertcat(X, Y, TH, W)

    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    th_init = ca.SX.sym('th_init')
    x_ref = ca.SX.sym('x_ref')
    y_ref = ca.SX.sym('y_ref')
    param = ca.vertcat(x_init, y_init, th_init, x_ref, y_ref)

    g = []
    obj = 0

    # Initial condition
    g.append(X[0] - x_init)
    g.append(Y[0] - y_init)
    g.append(TH[0] - th_init)

    # Dynamics constraints (nominal, no disturbance in planning)
    for k in range(N):
        x_next = X[k] + v * ca.cos(TH[k]) * dt
        y_next = Y[k] + v * ca.sin(TH[k]) * dt
        th_next = TH[k] + W[k] * dt
        g.append(X[k+1] - x_next)
        g.append(Y[k+1] - y_next)
        g.append(TH[k+1] - th_next)

    # Tightened obstacle constraints (increase radius by tube_margin)
    for k in range(N+1):
        for (ox, oy, r) in obstacles:
            tightened_r = r + margin + tube_margin
            g.append(((X[k] - ox)**2 + (Y[k] - oy)**2) - tightened_r**2)

    # Objective: final state to goal and control effort
    obj += (X[N] - x_ref)**2 + (Y[N] - y_ref)**2
    for k in range(N):
        obj += 0.01 * (W[k])**2

    nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': param}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    lbg = []
    ubg = []
    # Initial condition constraints
    for _ in range(3):
        lbg.append(0)
        ubg.append(0)
    # Dynamics constraints (3 per step)
    for _ in range(3 * N):
        lbg.append(0)
        ubg.append(0)
    # Tightened obstacle constraints
    for _ in range((N+1) * len(obstacles)):
        lbg.append(0)
        ubg.append(ca.inf)

    vars_init = np.zeros(((N+1)*3 + N,))
    lbx = []
    ubx = []
    # X, Y, TH unbounded
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    # W bounded
    for _ in range(N):
        lbx.append(-w_max)
        ubx.append(w_max)

    p_val = np.array([x0, y0, th0, x_goal, y_goal])
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val, x0=vars_init)
    sol_w = sol['x'][(N+1)*3:(N+1)*3+N].full().flatten()
    return sol_w[0]

if __name__ == "__main__":
    # Obstacles: (x_center, y_center, radius)
    obstacles = [
        (2.0, 2.0, 0.5),
        (4.0, 1.0, 0.5),
        (3.0, 3.0, 0.5)
    ]

    dt = 0.1
    N_mpc = 10
    v = 1.0
    w_max = 1.0
    max_iter = 200
    goal_tolerance = 0.1
    d_max = 0.15  # disturbance bound for simulation

    # For tube-based MPC, choose a tube_margin (can be tuned)
    tube_margin = 0.15

    # Start and goal
    x0, y0, th0 = 0.0, 0.0, 0.0
    x_goal, y_goal = 5.0, 4.0

    # Histories for standard MPC and tube-based robust MPC
    x_hist = [x0]
    y_hist = [y0]
    th_hist = [th0]

    x_hist_tube = [x0]
    y_hist_tube = [y0]
    th_hist_tube = [th0]

    x_curr, y_curr, th_curr = x0, y0, th0
    x_curr_tube, y_curr_tube, th_curr_tube = x0, y0, th0

    mpc_iter = 0
    tube_iter = 0

    while mpc_iter < max_iter and np.sqrt((x_curr - x_goal)**2 + (y_curr - y_goal)**2) > goal_tolerance:
        w_opt = dubins_mpc_step(x_curr, y_curr, th_curr, x_goal, y_goal, obstacles,
                                N=N_mpc, dt=dt, v=v, w_max=w_max)
        d_x = np.random.uniform(-d_max, d_max)
        d_y = np.random.uniform(-d_max, d_max)
        x_next = x_curr + v * np.cos(th_curr) * dt + d_x * dt
        y_next = y_curr + v * np.sin(th_curr) * dt + d_y * dt
        th_next = th_curr + w_opt * dt
        x_curr, y_curr, th_curr = x_next, y_next, th_next
        x_hist.append(x_curr)
        y_hist.append(y_curr)
        th_hist.append(th_curr)
        mpc_iter += 1

    while tube_iter < max_iter and np.sqrt((x_curr_tube - x_goal)**2 + (y_curr_tube - y_goal)**2) > goal_tolerance:
        w_opt_tube = tube_based_robust_mpc_step(x_curr_tube, y_curr_tube, th_curr_tube,
                                                 x_goal, y_goal, obstacles,
                                                 N=N_mpc, dt=dt, v=v, w_max=w_max,
                                                 margin=0.3, d_max=d_max, tube_margin=tube_margin)
        # Apply the control and add a random disturbance within bounds
        d_x = np.random.uniform(-d_max, d_max)
        d_y = np.random.uniform(-d_max, d_max)
        x_next_tube = x_curr_tube + v * np.cos(th_curr_tube) * dt + d_x * dt
        y_next_tube = y_curr_tube + v * np.sin(th_curr_tube) * dt + d_y * dt
        th_next_tube = th_curr_tube + w_opt_tube * dt
        x_curr_tube, y_curr_tube, th_curr_tube = x_next_tube, y_next_tube, th_next_tube
        x_hist_tube.append(x_curr_tube)
        y_hist_tube.append(y_curr_tube)
        th_hist_tube.append(th_curr_tube)
        tube_iter += 1

    plt.figure(figsize=(8, 8))
    plt.plot(x_hist, y_hist, 'b-', label='Standard MPC')
    plt.plot(x_hist_tube, y_hist_tube, 'g-', label='Tube-based Robust MPC')
    plt.plot(x0, y0, 'ko', label='Start')
    plt.plot(x_goal, y_goal, 'rx', markersize=10, label='Goal')
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r, color='r', fill=True, alpha=0.3)
        plt.gca().add_patch(circ)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Dubins Car: Standard MPC vs. Tube-based Robust MPC')
    plt.show()
