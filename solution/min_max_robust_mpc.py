import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

def dubins_mpc_step(x0, y0, th0, x_goal, y_goal, obstacles, N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3):
    """
    Standard MPC for the Dubins car.
    """
    # States: X, Y, TH (N+1 points) and control: W (N points)
    X = ca.SX.sym('X', N+1)
    Y = ca.SX.sym('Y', N+1)
    TH = ca.SX.sym('TH', N+1)
    W = ca.SX.sym('W', N)
    
    # Decision variables
    opt_vars = ca.vertcat(X, Y, TH, W)
    
    # Parameters: initial state and goal
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
    
    # Dynamics constraints
    for k in range(N):
        x_next = X[k] + v * ca.cos(TH[k]) * dt
        y_next = Y[k] + v * ca.sin(TH[k]) * dt
        th_next = TH[k] + W[k] * dt
        g.append(X[k+1] - x_next)
        g.append(Y[k+1] - y_next)
        g.append(TH[k+1] - th_next)
    
    # Obstacle avoidance constraints for each predicted state
    for k in range(N+1):
        for (ox, oy, r) in obstacles:
            g.append(((X[k] - ox)**2 + (Y[k] - oy)**2) - (r + margin)**2)
    
    # Objective: final distance to goal + small control penalty
    obj += (X[N] - x_ref)**2 + (Y[N] - y_ref)**2
    for k in range(N):
        obj += 0.01 * (W[k])**2

    # Create NLP
    nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': param}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    
    # Bounds for constraints
    lbg = []
    ubg = []
    # Initial condition constraints (3)
    for _ in range(3):
        lbg.append(0)
        ubg.append(0)
    # Dynamics constraints (3*N)
    for _ in range(3*N):
        lbg.append(0)
        ubg.append(0)
    # Obstacle constraints ((N+1)*len(obstacles)): must be >= 0
    for _ in range((N+1) * len(obstacles)):
        lbg.append(0)
        ubg.append(ca.inf)
    
    # Decision variable bounds
    vars_init = np.zeros(((N+1)*3 + N, ))
    lbx = []
    ubx = []
    # X, Y, TH are unbounded
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    for _ in range(N+1):
        lbx.append(-ca.inf)
        ubx.append(ca.inf)
    # W bounded in [-w_max, w_max]
    for _ in range(N):
        lbx.append(-w_max)
        ubx.append(w_max)
    
    # Solve
    p_val = np.array([x0, y0, th0, x_goal, y_goal])
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val, x0=vars_init)
    sol_w = sol['x'][(N+1)*3:(N+1)*3+N].full().flatten()
    return sol_w[0]

def robust_dubins_mpc_step(x0, y0, th0, x_goal, y_goal, obstacles, N=10, dt=0.1, v=1.0, w_max=1.0, margin=0.3, d_max=0.1):
    """
    Robust MPC for the Dubins car using a min-max formulation over a set of 4 disturbance scenarios.
    The disturbance is assumed to be constant over the horizon for each scenario.
    Scenarios: (d_x, d_y) in {(d_max, d_max), (d_max, -d_max), (-d_max, d_max), (-d_max, -d_max)}.
    The control sequence is common across scenarios.
    """
    scenarios = [(d_max, d_max), (d_max, -d_max), (-d_max, d_max), (-d_max, -d_max)]
    n_scen = len(scenarios)
    
    # For each scenario, define states: X_s, Y_s, TH_s (each of length N+1)
    X = [ca.SX.sym(f'X_{s}', N+1) for s in range(n_scen)]
    Y = [ca.SX.sym(f'Y_{s}', N+1) for s in range(n_scen)]
    TH = [ca.SX.sym(f'TH_{s}', N+1) for s in range(n_scen)]
    # Common control sequence
    W = ca.SX.sym('W', N)
    # Auxiliary variable for worst-case cost
    gamma = ca.SX.sym('gamma')
    
    # Collect decision variables
    decision_vars = []
    for s in range(n_scen):
        decision_vars.append(X[s])
        decision_vars.append(Y[s])
        decision_vars.append(TH[s])
    decision_vars.append(W)
    decision_vars.append(gamma)
    opt_vars = ca.vertcat(*decision_vars)
    
    # Parameters: initial state and goal (common to all scenarios)
    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    th_init = ca.SX.sym('th_init')
    x_ref = ca.SX.sym('x_ref')
    y_ref = ca.SX.sym('y_ref')
    param = ca.vertcat(x_init, y_init, th_init, x_ref, y_ref)
    
    g = []
    obj_constr = []
    
    # For each scenario, add dynamics and obstacle constraints, and relate final cost to gamma.
    for s in range(n_scen):
        d_x, d_y = scenarios[s]
        # Initial condition
        g.append(X[s][0] - x_init)
        g.append(Y[s][0] - y_init)
        g.append(TH[s][0] - th_init)
        # Dynamics with disturbance
        for k in range(N):
            x_next = X[s][k] + v * ca.cos(TH[s][k]) * dt + d_x * dt
            y_next = Y[s][k] + v * ca.sin(TH[s][k]) * dt + d_y * dt
            th_next = TH[s][k] + W[k] * dt
            g.append(X[s][k+1] - x_next)
            g.append(Y[s][k+1] - y_next)
            g.append(TH[s][k+1] - th_next)
        # Obstacle constraints for each predicted state
        for k in range(N+1):
            for (ox, oy, r) in obstacles:
                g.append(((X[s][k] - ox)**2 + (Y[s][k] - oy)**2) - (r + margin)**2)
        # Final cost for scenario s
        cost_s = (X[s][N] - x_ref)**2 + (Y[s][N] - y_ref)**2
        for k in range(N):
            cost_s += 0.01 * (W[k])**2
        # Enforce gamma >= cost for each scenario
        obj_constr.append(gamma - cost_s)
    
    # Overall objective: minimize gamma (the worst-case cost)
    obj = gamma
    
    # Create NLP
    g_all = ca.vertcat(*g, *obj_constr)
    nlp = {'x': opt_vars, 'f': obj, 'g': g_all, 'p': param}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    
    # Count constraints to set bounds.
    lbg = []
    ubg = []
    # For each scenario: initial conditions (3 constraints) and dynamics (3*N constraints)
    for s in range(n_scen):
        # initial conditions: 3
        for _ in range(3):
            lbg.append(0)
            ubg.append(0)
        # dynamics: 3*N
        for _ in range(3 * N):
            lbg.append(0)
            ubg.append(0)
        # obstacle constraints: (N+1)*len(obstacles)
        for _ in range((N+1) * len(obstacles)):
            lbg.append(0)
            ubg.append(ca.inf)
    # Constraints for gamma >= cost for each scenario (n_scen constraints)
    for _ in range(n_scen):
        lbg.append(0)
        ubg.append(ca.inf)
    
    # Variable bounds and initial guess
    vars_init = np.zeros(((n_scen * (N+1) * 3) + N + 1,))
    lbx = []
    ubx = []
    # For each scenario, X, Y, TH unbounded
    for s in range(n_scen):
        for _ in range(N+1):
            lbx.append(-ca.inf)
            ubx.append(ca.inf)
        for _ in range(N+1):
            lbx.append(-ca.inf)
            ubx.append(ca.inf)
        for _ in range(N+1):
            lbx.append(-ca.inf)
            ubx.append(ca.inf)
    # W: control variables bounded
    for _ in range(N):
        lbx.append(-w_max)
        ubx.append(w_max)
    # gamma: unbounded from below (we minimize it)
    lbx.append(-ca.inf)
    ubx.append(ca.inf)
    
    p_val = np.array([x0, y0, th0, x_goal, y_goal])
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val, x0=vars_init)
    # Extract control sequence (W are after the state variables)
    offset = n_scen * (N+1) * 3
    sol_W = sol['x'][offset:offset+N].full().flatten()
    return sol_W[0]

if __name__ == "__main__":
    # Obstacles: list of (x_center, y_center, radius)
    obstacles = [
        (2.0, 2.0, 0.5),
        (4.0, 1.0, 0.5),
        (3.0, 3.0, 0.5)
    ]
    
    # Simulation parameters
    dt = 0.1
    N_mpc = 10
    v = 1.0
    w_max = 1.0
    max_iter = 200
    goal_tolerance = 0.1
    d_max = 0.3  # disturbance magnitude for robust MPC
    
    # Initial state and goal
    x0, y0, th0 = 0.0, 0.0, 0.0
    x_goal, y_goal = 5.0, 4.0
    
    # Histories for standard MPC
    x_hist = [x0]
    y_hist = [y0]
    th_hist = [th0]
    
    # Histories for robust MPC
    x_hist_r = [x0]
    y_hist_r = [y0]
    th_hist_r = [th0]
    
    x_curr, y_curr, th_curr = x0, y0, th0
    x_curr_r, y_curr_r, th_curr_r = x0, y0, th0
    
    mpc_iter = 0
    robust_iter = 0
    while mpc_iter < max_iter and np.sqrt((x_curr - x_goal)**2 + (y_curr - y_goal)**2) > goal_tolerance:
        # Standard MPC step (no disturbance)
        w_opt = dubins_mpc_step(x_curr, y_curr, th_curr,
                                x_goal, y_goal,
                                obstacles,
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

    while robust_iter < max_iter and np.sqrt((x_curr_r - x_goal)**2 + (y_curr_r - y_goal)**2) > goal_tolerance:
        # Robust MPC step (simulate with random disturbance in [-d_max, d_max])
        w_opt_r = robust_dubins_mpc_step(x_curr_r, y_curr_r, th_curr_r,
                                         x_goal, y_goal,
                                         obstacles,
                                         N=N_mpc, dt=dt, v=v, w_max=w_max, d_max=d_max)
        # Simulate worst-case-like disturbance for robust MPC:
        # For simulation, we draw a random disturbance within the bounds.
        d_x = np.random.uniform(-d_max, d_max)
        d_y = np.random.uniform(-d_max, d_max)
        x_next_r = x_curr_r + v * np.cos(th_curr_r) * dt + d_x * dt
        y_next_r = y_curr_r + v * np.sin(th_curr_r) * dt + d_y * dt
        th_next_r = th_curr_r + w_opt_r * dt
        x_curr_r, y_curr_r, th_curr_r = x_next_r, y_next_r, th_next_r
        x_hist_r.append(x_curr_r)
        y_hist_r.append(y_curr_r)
        th_hist_r.append(th_curr_r)
        robust_iter += 1

    # Plot trajectories
    plt.figure(figsize=(8, 8))
    plt.plot(x_hist, y_hist, 'b-', label='Standard MPC')
    plt.plot(x_hist_r, y_hist_r, 'g-', label='Robust MPC')
    plt.plot(x0, y0, 'ko', label='Start')
    plt.plot(x_goal, y_goal, 'rx', markersize=10, label='Goal')
    # Plot obstacles
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r, color='r', fill=True, alpha=0.3)
        plt.gca().add_patch(circ)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Dubins Car: Standard MPC vs. Robust MPC')
    plt.show()
