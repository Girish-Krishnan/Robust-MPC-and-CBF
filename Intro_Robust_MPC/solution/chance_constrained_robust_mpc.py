import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

def dubins_mpc_step(x0, y0, th0, x_goal, y_goal, obstacles,
                    N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3):
    """
    Standard MPC for the Dubins car.
    """
    # Define decision variables: states (X, Y, TH) and control (W)
    X = ca.SX.sym('X', N+1)
    Y = ca.SX.sym('Y', N+1)
    TH = ca.SX.sym('TH', N+1)
    W = ca.SX.sym('W', N)

    opt_vars = ca.vertcat(X, Y, TH, W)

    # Parameters: initial state and goal
    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    th_init = ca.SX.sym('th_init')
    x_ref  = ca.SX.sym('x_ref')
    y_ref  = ca.SX.sym('y_ref')
    param  = ca.vertcat(x_init, y_init, th_init, x_ref, y_ref)

    g = []  # constraints
    obj = 0  # cost

    # Initial condition
    g.append(X[0] - x_init)
    g.append(Y[0] - y_init)
    g.append(TH[0] - th_init)

    # Nominal dynamics constraints
    for k in range(N):
        x_next = X[k] + v * ca.cos(TH[k]) * dt
        y_next = Y[k] + v * ca.sin(TH[k]) * dt
        th_next = TH[k] + W[k] * dt
        g.append(X[k+1] - x_next)
        g.append(Y[k+1] - y_next)
        g.append(TH[k+1] - th_next)

    # Obstacle avoidance constraints (no tightening)
    for k in range(N+1):
        for (ox, oy, r) in obstacles:
            g.append(((X[k] - ox)**2 + (Y[k] - oy)**2) - (r + margin)**2)

    # Objective: final distance to goal and control effort
    obj += (X[N] - x_ref)**2 + (Y[N] - y_ref)**2
    for k in range(N):
        obj += 0.01 * (W[k])**2

    nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': param}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Constraint bounds
    lbg = []
    ubg = []
    # initial state: 3 constraints
    for _ in range(3):
        lbg.append(0)
        ubg.append(0)
    # dynamics: 3*N constraints
    for _ in range(3*N):
        lbg.append(0)
        ubg.append(0)
    # obstacles: (N+1)*len(obstacles)
    for _ in range((N+1)*len(obstacles)):
        lbg.append(0)
        ubg.append(ca.inf)

    # Decision variable bounds and initial guess
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
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                 p=p_val, x0=vars_init)
    sol_w = sol['x'][(N+1)*3:(N+1)*3+N].full().flatten()
    return sol_w[0]

def chance_constrained_robust_mpc_step(x0, y0, th0, x_goal, y_goal, obstacles,
                                       N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3,
                                       sigma=0.1, prob_level=0.95):
    """
    Stochastic MPC using chance constraints.
    Assumes additive Gaussian disturbances with standard deviation sigma.
    The chance constraint is enforced by tightening the obstacle constraints.
    For a 2D Gaussian, the tightening radius is increased by a factor k,
    where k = sqrt(chi2.ppf(prob_level, df=2)). For example, for prob_level=0.95,
    k ~ 2.4477.
    """
    # Compute tightening factor
    # For 2 degrees of freedom, using chi2 quantile:
    # k = sqrt(chi2.ppf(prob_level, df=2))
    # For simplicity, we use a precomputed value for 95% probability.
    if prob_level == 0.95:
        k = 2.4477
    else:
        # Otherwise, approximate with inverse error function
        from math import sqrt
        k = np.sqrt(-2 * np.log(1 - prob_level))
    
    # Safety margin due to disturbance uncertainty
    safety_margin = k * sigma

    # Define decision variables (nominal trajectory)
    X = ca.SX.sym('X', N+1)
    Y = ca.SX.sym('Y', N+1)
    TH = ca.SX.sym('TH', N+1)
    W = ca.SX.sym('W', N)

    opt_vars = ca.vertcat(X, Y, TH, W)

    # Parameters: initial state and goal
    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    th_init = ca.SX.sym('th_init')
    x_ref  = ca.SX.sym('x_ref')
    y_ref  = ca.SX.sym('y_ref')
    param  = ca.vertcat(x_init, y_init, th_init, x_ref, y_ref)

    g = []  # constraints
    obj = 0  # objective

    # Initial condition
    g.append(X[0] - x_init)
    g.append(Y[0] - y_init)
    g.append(TH[0] - th_init)

    # Nominal dynamics (no disturbance in planning)
    for k in range(N):
        x_next = X[k] + v * ca.cos(TH[k]) * dt
        y_next = Y[k] + v * ca.sin(TH[k]) * dt
        th_next = TH[k] + W[k] * dt
        g.append(X[k+1] - x_next)
        g.append(Y[k+1] - y_next)
        g.append(TH[k+1] - th_next)

    # Tightened obstacle constraints: increase effective radius
    # by safety_margin to account for disturbance uncertainty.
    for k in range(N+1):
        for (ox, oy, r) in obstacles:
            tightened_r = r + margin + safety_margin
            g.append(((X[k] - ox)**2 + (Y[k] - oy)**2) - tightened_r**2)

    # Objective: final distance to goal and control effort
    obj += (X[N] - x_ref)**2 + (Y[N] - y_ref)**2
    for k in range(N):
        obj += 0.01 * (W[k])**2

    nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': param}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Constraint bounds
    lbg = []
    ubg = []
    # Initial condition: 3 constraints
    for _ in range(3):
        lbg.append(0)
        ubg.append(0)
    # Dynamics constraints: 3 per step
    for _ in range(3*N):
        lbg.append(0)
        ubg.append(0)
    # Tightened obstacle constraints: (N+1)*len(obstacles)
    for _ in range((N+1)*len(obstacles)):
        lbg.append(0)
        ubg.append(ca.inf)

    # Decision variable bounds and initial guess
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
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                 p=p_val, x0=vars_init)
    sol_w = sol['x'][(N+1)*3:(N+1)*3+N].full().flatten()
    return sol_w[0]

if __name__ == "__main__":
    # Define obstacles as (x_center, y_center, radius)
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

    # Parameters for chance-constrained MPC
    sigma = 0.1       # standard deviation of additive Gaussian disturbance
    prob_level = 0.9 # desired probability level for constraint satisfaction

    # Initial state and goal
    x0, y0, th0 = 0.0, 0.0, 0.0
    x_goal, y_goal = 5.0, 4.0

    # Histories for standard MPC and chance-constrained robust MPC
    x_hist = [x0]
    y_hist = [y0]
    th_hist = [th0]

    x_hist_cc = [x0]
    y_hist_cc = [y0]
    th_hist_cc = [th0]

    x_curr, y_curr, th_curr = x0, y0, th0
    x_curr_cc, y_curr_cc, th_curr_cc = x0, y0, th0

    mpc_iter = 0
    cc_iter = 0

    # Run standard MPC simulation (without disturbance)
    while mpc_iter < max_iter and np.sqrt((x_curr - x_goal)**2 + (y_curr - y_goal)**2) > goal_tolerance:
        w_opt = dubins_mpc_step(x_curr, y_curr, th_curr,
                                x_goal, y_goal, obstacles,
                                N=N_mpc, dt=dt, v=v, w_max=w_max)
        x_next = x_curr + v * np.cos(th_curr) * dt
        y_next = y_curr + v * np.sin(th_curr) * dt
        th_next = th_curr + w_opt * dt
        x_curr, y_curr, th_curr = x_next, y_next, th_next
        x_hist.append(x_curr)
        y_hist.append(y_curr)
        th_hist.append(th_curr)
        mpc_iter += 1

    # Run chance-constrained robust MPC simulation (simulate with additive Gaussian disturbance)
    while cc_iter < max_iter and np.sqrt((x_curr_cc - x_goal)**2 + (y_curr_cc - y_goal)**2) > goal_tolerance:
        w_opt_cc = chance_constrained_robust_mpc_step(x_curr_cc, y_curr_cc, th_curr_cc,
                                                      x_goal, y_goal, obstacles,
                                                      N=N_mpc, dt=dt, v=v, w_max=w_max,
                                                      margin=0.3, sigma=sigma, prob_level=prob_level)
        # Simulate the system with additive Gaussian disturbance
        d_x = np.random.normal(0, sigma)
        d_y = np.random.normal(0, sigma)
        x_next_cc = x_curr_cc + v * np.cos(th_curr_cc) * dt + d_x * dt
        y_next_cc = y_curr_cc + v * np.sin(th_curr_cc) * dt + d_y * dt
        th_next_cc = th_curr_cc + w_opt_cc * dt
        x_curr_cc, y_curr_cc, th_curr_cc = x_next_cc, y_next_cc, th_next_cc
        x_hist_cc.append(x_curr_cc)
        y_hist_cc.append(y_curr_cc)
        th_hist_cc.append(th_curr_cc)
        cc_iter += 1

    # Plot trajectories
    plt.figure(figsize=(8,8))
    plt.plot(x_hist, y_hist, 'b-', label='Standard MPC')
    plt.plot(x_hist_cc, y_hist_cc, 'g-', label='Chance-Constrained Robust MPC')
    plt.plot(x0, y0, 'ko', label='Start')
    plt.plot(x_goal, y_goal, 'rx', markersize=10, label='Goal')
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r, color='r', fill=True, alpha=0.3)
        plt.gca().add_patch(circ)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Dubins Car: Standard MPC vs. Chance-Constrained Robust MPC')
    plt.show()
