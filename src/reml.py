import torch
import numpy as np
from scipy.optimize import root_scalar

def dQ_dt(t, lambda_vals, c_squared, n):
    exp_term = np.exp(-t * lambda_vals)
    numerator = np.sum(lambda_vals * c_squared * exp_term)
    denominator = np.sum(c_squared * exp_term)
    if denominator == 0: return np.inf
    return - n * numerator / denominator + np.sum(lambda_vals)

def compute_reml_t(H, y_train, f0_train, learning_rate):
    """
    Calculates the optimal stopping time t_reml.
    """
    n_train = H.shape[0]
    
    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(H)
    
    # Prepare terms
    u0 = (y_train - f0_train).detach().numpy()
    eigvals = eigvals.detach().numpy().reshape(n_train, 1)
    eigvecs = eigvecs.detach().numpy()
    c_squared = ((eigvecs.T @ u0)**2)

    # Solve root
    try:
        t_sol = root_scalar(dQ_dt, args=(eigvals, c_squared, n_train), 
                            bracket=[0, 100000], method='bisect') # Wide bracket for safety
        t_reml = t_sol.root / learning_rate
        return t_reml
    except Exception as e:
        print(f"Warning: REML root finding failed: {e}")
        return 0