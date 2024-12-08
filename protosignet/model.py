import numpy as np
from scipy.integrate import ode


def sim_signet(tt, uu, kr, ku, kX):
    """Simulates the dynamic response of the SigNet ODE model with N nodes.

    Args:
        tt (1 x t array): t equally spaced timepoints
        uu (1 x t array): input/stimuli amount at each timepoint
        kr (1 x N array): reversion rates of each node
        ku (1 x N array): induction rates of stimuli towards each node
        kX (N x N array): interaction rates of each node towards each other

    Returns:
        sol_t (1D array): timepoints of the simulation (same as tt)
        sol_X (2D array): N x t array of the dynamic response for each node
    """
    # reversion rates dictate whether the initial value of a node is 0 or 1
    X0 = np.where(kr < 0, 0, 1)
    # induction direction is always opposite of reversion direction
    ku = np.where(kr * ku > 0, -ku, ku)

    def model(t, X):
        # reaction towards a target node depends on that node's current amount
        Xr = np.where(kr < 0, X, 1 - X)
        Xu = np.where(ku < 0, X, 1 - X)
        Xy = np.where(kX.T < 0, X, 1 - X).T
        # dx/dt = reversion + induction + interaction
        dX = kr * Xr + ku * Xu * u + np.sum(kX * Xy * X, axis=1)
        return dX

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5, max_step=0.1)
    solver.set_initial_value(X0)
    sol_t = [tt[0]]
    sol_X = [X0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_X.append(solver.y)
    return np.array(sol_t), np.array(sol_X).T
