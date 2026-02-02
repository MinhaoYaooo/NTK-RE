import torch
from functorch import make_functional, vmap, jacrev

def get_ntk_matrices(model, X_train, X_test):
    """
    Computes NTK matrices H (Train-Train) and C (Test-Train).
    """
    # Create functional version of the model
    fnet, params = make_functional(model)

    def fnet_single(params, z):
        return fnet(params, z.unsqueeze(0)).squeeze(0)

    def empirical_ntk_jacobian_contraction(params, z1, z2):
        # Compute J(z1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, z1)
        jac1 = [j.flatten(2) for j in jac1]
        
        # Compute J(z2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, z2)
        jac2 = [j.flatten(2) for j in jac2]
        
        # Compute J(z1) @ J(z2).T
        result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result

    # Compute H (Train-Train)
    n_train = X_train.shape[0]
    result_H = empirical_ntk_jacobian_contraction(params, X_train, X_train)
    H = result_H[:,:,0,0] / n_train

    # Compute C (Test-Train)
    result_C = empirical_ntk_jacobian_contraction(params, X_test, X_train)
    C = result_C[:,:,0,0] / n_train

    return H, C, fnet, params