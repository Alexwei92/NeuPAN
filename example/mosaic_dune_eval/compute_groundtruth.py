import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from neupan.blocks.dune_train import DUNETrain


def _to_numpy_col(p: torch.Tensor) -> np.ndarray:
    """Convert a (2,) or (2,1) torch or numpy array to shape (2,1) numpy array.
    """
    if isinstance(p, torch.Tensor):
        return p.detach().cpu().numpy().reshape(2, 1)
    else:
        return np.asarray(p).reshape(2, 1)


def compute_ground_truth(
    G,
    h,
    obs_points,
    robot_state: Optional[torch.Tensor] = None,
    R_override: Optional[object] = None,
    checkpoint_path: str = "/tmp",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[np.ndarray]]:
    """
    Compute ground-truth mu, lambda and distance for a set of obstacle points using the convex
    solver wrapped by `DUNETrain.prob_solve`.

    Args:
        G: robot G matrix (torch.Tensor or numpy.ndarray) with shape (edge_dim, state_dim)
        h: robot h vector (torch.Tensor or numpy.ndarray) with shape (edge_dim, 1)
        obs_points: points in WORLD coordinates as torch.Tensor or numpy.ndarray with shape (2, N)
           OR points already in ROBOT coordinates if robot_state and R_override are None
        robot_state: OPTIONAL robot state [x, y, theta] as torch.Tensor or numpy.ndarray
           for transforming obstacle points from world to robot frame
           If None, obs_points are assumed to already be in robot frame
        R_override: optional rotation matrix override, if not using robot_state's theta
        checkpoint_path: path passed to DUNETrain (unused here except for interface).

    Returns:
        mu_list: list of torch.Tensor, each shape (edge_dim, 1)
        lam_list: list of torch.Tensor, each shape (state_dim, 1)
        distance_list: list of floats
        p0_list: list of point coordinates in robot frame (2,1) as numpy arrays
    """

    # Ensure obs_points is iterable of columns
    if obs_points is None:
        return [], [], [], []

    # Instantiate DUNETrain with a dummy model (model not used for prob_solve)
    # DUNETrain expects G and h in the same types used in the repo (torch tensors are fine).
    # DUNETrain expects a model argument with a .parameters() method for the optimizer.
    # Provide a minimal torch.nn.Module instance that contains at least one parameter
    # so the optimizer can be constructed (avoid "empty parameter list" error).
    dummy_model = torch.nn.Linear(1, 1)
    # Convert G and h to float32 tensors to avoid dtype mismatches later.
    # Get device information from input tensors - prioritize R_override's device
    device = None
    
    # First check R_override since it comes from PAN's R_list
    if R_override is not None and isinstance(R_override, torch.Tensor) and hasattr(R_override, 'device'):
        device = R_override.device
    # Then check robot_state
    elif isinstance(robot_state, torch.Tensor) and hasattr(robot_state, 'device'):
        device = robot_state.device
    # Then check G
    elif isinstance(G, torch.Tensor) and hasattr(G, 'device'):
        device = G.device
    
    # Default to CPU if no device is detected
    if device is None:
        device = torch.device('cpu')
        
    print(f"Using device {device} for ground truth computation")
    
    # Convert G and h to float32 tensors with correct device
    if not isinstance(G, torch.Tensor):
        G_t = torch.as_tensor(G, dtype=torch.float32, device=device)
    else:
        G_t = G.to(dtype=torch.float32, device=device)

    if not isinstance(h, torch.Tensor):
        h_t = torch.as_tensor(h, dtype=torch.float32, device=device)
    else:
        h_t = h.to(dtype=torch.float32, device=device)

    # Move dummy model to the device we're using
    dummy_model = dummy_model.to(device)
        
    try:
        dt = DUNETrain(dummy_model, G_t, h_t, checkpoint_path)
    except Exception as e:
        # If DUNETrain still fails for some reason, re-raise with context
        raise RuntimeError(f"DUNETrain initialization failed: {e} (G device: {G_t.device}, h device: {h_t.device}, model device: {next(dummy_model.parameters()).device}")

    # Transform points to robot frame if needed, or use provided points directly
    # Normalize obs_points into torch tensor and move to our device
    if isinstance(obs_points, torch.Tensor):
        obs_t = obs_points.to(device=device, dtype=torch.float32)
    else:
        obs_t = torch.as_tensor(obs_points, dtype=torch.float32, device=device)

    # Ensure shape is (2, N)
    if obs_t.ndim == 1:
        obs_t = obs_t.reshape(2, 1)
    elif obs_t.ndim == 2 and obs_t.shape[0] != 2:
        if obs_t.shape[1] == 2:
            obs_t = obs_t.t()  # Transpose if shape is (N, 2)
        else:
            raise ValueError("obs_points must be shape (2,N) or (N,2)")
    
    if robot_state is not None:
        # Transform points from world to robot frame
        if isinstance(robot_state, torch.Tensor):
            state = robot_state.reshape((3, 1)).to(device)
        else:
            state = np.asarray(robot_state).reshape((3, 1))
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
            
        # Exactly as in point_state_transform
        trans = state[0:2]
        theta = state[2, 0].item()  # Extract scalar value
        
        # Create rotation matrix R exactly as in PAN
        R_np = np.array([[np.cos(theta), -np.sin(theta)], 
                       [np.sin(theta), np.cos(theta)]])
        R_t = torch.as_tensor(R_np, dtype=torch.float32, device=device)
        
        # Override rotation if specified (e.g. using PAN's R_list[0])
        if R_override is not None:
            if isinstance(R_override, torch.Tensor):
                R_t = R_override.to(dtype=torch.float32, device=device)
            else:
                R_t = torch.as_tensor(R_override, dtype=torch.float32, device=device)
            print(f"Using R_override matrix on device {R_t.device}")
            
        # Double-check device consistency - crucial for operations
        print(f"Device check before transform - R_t: {R_t.device}, obs_t: {obs_t.device}, trans: {trans.device}")

        # Transform all points to robot frame at once - EXACTLY as in point_state_transform
        # Double check that all tensors are on the same device before matrix operation
        print(f"R_t device: {R_t.device}, obs_t device: {obs_t.device}, trans device: {trans.device}")
        
        if R_t.device != device:
            R_t = R_t.to(device)
        if obs_t.device != device:
            obs_t = obs_t.to(device)
        if trans.device != device:
            trans = trans.to(device)
            
        p0_t = R_t.t() @ (obs_t - trans)
    else:
        # If no robot_state, assume points are already in robot frame
        p0_t = obs_t
        # Still need R_t for lambda calculation
        if R_override is not None:
            if isinstance(R_override, torch.Tensor):
                R_t = R_override.to(dtype=torch.float32, device=device)
            else:
                R_t = torch.as_tensor(R_override, dtype=torch.float32, device=device)
            print(f"Using R_override matrix on device {R_t.device}")
        else:
            # Default identity rotation if neither robot_state nor R_override is provided
            R_t = torch.eye(2, dtype=torch.float32, device=device)
    
    # Convert to list of numpy arrays
    p0_list = [p0_t[:, i].reshape(2, 1).detach().cpu().numpy() for i in range(p0_t.shape[1])]

    mu_list = []
    lam_list = []
    distance_list = []

    # Process each point in robot frame
    for i in range(len(p0_list)):
        p = p0_list[i]  # p is numpy (2,1) in robot frame
        try:
            # Use CVXPY solver (ECOS) - same as in DUNE
            obj_value, mu_value = dt.prob_solve(p)
        except Exception as e:
            # propagate a helpful error
            raise RuntimeError(f"convex solver failed for point {p}: {e}")
            
        # mu_value is numpy array (edge_dim, 1) from the solver
        # Make sure all tensors stay on the consistent device
        mu_t = torch.as_tensor(mu_value, dtype=G_t.dtype, device=device)
        
        # Ensure mu follows the constraint ||G.T @ mu|| <= 1 exactly as in DUNE
        # This is essential to match the DUNE model's output scaling
        # Double check device consistency - all tensors MUST be on same device
        if G_t.device != device:
            G_t = G_t.to(device)
        if mu_t.device != device:
            mu_t = mu_t.to(device)
            
        G_t_mu = G_t.t() @ mu_t  # This operation requires both tensors on same device
        G_t_mu_norm = torch.norm(G_t_mu)
        if G_t_mu_norm > 1:
            # Rescale mu to satisfy the constraint exactly
            mu_t = mu_t / G_t_mu_norm
        
        # Compute distance exactly as in DUNE.cal_objective_distance:
        # distance = mu.T @ (G @ p0 - h)
        p_tensor = torch.as_tensor(p, dtype=G_t.dtype, device=device)
        
        # Double-check device consistency for all tensors
        if G_t.device != device:
            G_t = G_t.to(device)
        if h_t.device != device:
            h_t = h_t.to(device)
        
        objective = G_t @ p_tensor - h_t
        distance_value = float((mu_t.t() @ objective).item())
        distance_list.append(distance_value)
        
        # IMPORTANT: The lambda sign error is here!
        # In PAN's implementation:
        #    - R is defined as [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        #    - p0 = R.T @ (obs - trans)  <-- Note use of R.T
        #    - lam = -R @ G.T @ mu  <-- Note use of R (not R.T)
        
        # In our ground truth calculation:
        #    - R_t is defined same way as R in PAN: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        #    - p0_t = R_t.t() @ (obs_t - trans)  <-- Also using R_t.t()
        #    - So we should use R_t (not R_t.t()) for lambda
        
        if R_t.device != device:
            R_t = R_t.to(device)  # Ensure R_t is on the same device
        if G_t.device != device:
            G_t = G_t.to(device)
        if mu_t.device != device:
            mu_t = mu_t.to(device)
        
        # The key is: when PAN calls R @ G.T @ mu, our equivalent is R_t @ G_t.t() @ mu_t
        lam_t = -R_t @ G_t.t() @ mu_t
        
        # Normalize mu and lambda to match PAN's scaling
        # PAN uses a neural network that might produce differently scaled outputs
        # We need to normalize to comparable magnitudes
        mu_norm = torch.norm(mu_t)
        if mu_norm > 0:
            mu_t = mu_t / mu_norm  # Normalize mu to unit norm

        # Store results
        mu_list.append(mu_t)
        lam_list.append(lam_t)
    
    # Convert distance_list to tensor for sorting
    distance_tensor = torch.tensor(distance_list, device=device)
    
    # Sort all results by distance (exactly as done in DUNE)
    sort_indices = torch.argsort(distance_tensor)
    
    # Convert to list of numpy arrays or tensors (consistent with original return)
    sorted_mu_list = [mu_list[i] for i in sort_indices]
    sorted_lam_list = [lam_list[i] for i in sort_indices]
    sorted_distance_list = [distance_list[i] for i in sort_indices]
    sorted_p0_list = [p0_list[i] for i in sort_indices]
    
    return sorted_mu_list, sorted_lam_list, sorted_distance_list, sorted_p0_list


if __name__ == "__main__":
    # quick smoke test when run directly: create a dummy G,h and a point
    import numpy as _np

    G = _np.array([[1.0, 0.0], [0.0, 1.0]])  # simple constraints
    h = _np.array([[0.5], [0.5]])
    pts = _np.array([[0.0, 0.6], [0.0, 0.6]])

    mu_list, lam_list, dist_list, p0_list = compute_ground_truth(G, h, pts)
    print('computed', len(mu_list), 'points. First distance:', dist_list[0])