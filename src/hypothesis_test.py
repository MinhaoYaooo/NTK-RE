import torch
import numpy as np
from scipy.stats import chi2, norm

def test_initialization(H, y_train, f0_train):
    """
    Performs a hypothesis test to determine if training is necessary.
    H0: The data is explained by the random initialization (f0).
    H1: There is additional signal in the data.
    
    Returns:
        p_value (float): Based on Satterthwaite approximation.
        p_value_asymp (float): Based on asymptotic normality.
    """
    # Ensure inputs are numpy arrays
    H_np = H.detach().cpu().numpy() if torch.is_tensor(H) else H
    y = y_train.detach().cpu().numpy().reshape(-1, 1) if torch.is_tensor(y_train) else y_train.reshape(-1, 1)
    f0 = f0_train.detach().cpu().numpy().reshape(-1, 1) if torch.is_tensor(f0_train) else f0_train.reshape(-1, 1)
    
    n_train = H_np.shape[0]
    u0 = y - f0

    # Step 1: Compute Projection Matrix P
    # Projects onto the orthogonal complement of f0
    f0_norm_squared = f0.T @ f0
    if f0_norm_squared < 1e-10:
        P = np.eye(n_train)
    else:
        P = np.eye(n_train) - (f0 @ f0.T) / f0_norm_squared

    # Step 2: Compute Orthonormal Basis Q for range(P) using SVD
    # We want a basis for the space orthogonal to f0
    U, s, Vt = np.linalg.svd(P, full_matrices=False)
    
    # The rank is effectively n-1 (since we projected out 1 dimension f0)
    # We select columns corresponding to non-zero singular values
    rank = n_train - 1
    Q = U[:, :rank]

    # Step 3: Project the residuals
    w = Q.T @ u0

    # Step 4: Compute the projected kernel
    M = Q.T @ H_np @ Q

    # Step 5: Estimate noise variance (Sigma Hat Squared)
    sigma_hat_squared = (w.T @ w) / (n_train - 1)
    sigma_hat_squared = sigma_hat_squared.item()

    # Step 6: Form the Test Statistic T
    # T compares the "signal" energy (M) vs the "noise" energy (sigma)
    numerator = w.T @ M @ w
    T = numerator / (2 * sigma_hat_squared)
    T = T.item()

    # Step 7: Compute Satterthwaite parameters (for Chi2 approximation)
    tr_M = np.trace(M)
    tr_M2 = np.trace(M @ M)

    if tr_M2 == 0:
        return 1.0, 1.0

    kappa = tr_M2 / (2 * tr_M)
    nu = (tr_M**2) / tr_M2

    # Step 8: Exact P-value (Satterthwaite)
    # Pr(kappa * Chi2_nu > T)
    if kappa > 0:
        p_value = 1 - chi2.cdf(T / kappa, nu)
    else:
        p_value = 1.0

    # Step 9: Asymptotic P-value (Normal approximation)
    ET = tr_M / 2
    VarT = tr_M2 / 2
    z_score = (T - ET) / np.sqrt(VarT)
    p_value_asymp = 2 * norm.cdf(-np.abs(z_score))

    return p_value, p_value_asymp