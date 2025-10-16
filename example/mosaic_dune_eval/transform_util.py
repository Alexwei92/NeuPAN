import numpy as np
import torch

def transform_points_to_robot_frame(points_world, robot_state):
    """
    Transform points from world frame to robot frame.
    
    Args:
        points_world: Points in world frame (torch.Tensor)
            Shape can be (2,), (2,1), (2,N), or (2,N,1)
        robot_state: Robot state tensor [x, y, theta] (torch.Tensor)
    
    Returns:
        points_robot: Points in robot frame (torch.Tensor)
            Will have same shape as points_world
    """
    # Handle different input shapes
    original_shape = points_world.shape
    
    # Extract robot position and orientation
    robot_pos = robot_state[:2].reshape(2, 1)
    theta = robot_state[2].item()
    
    # Reshape points_world to (2,N) for easier processing
    if len(original_shape) == 1:  # (2,)
        points = points_world.reshape(2, 1)
    elif len(original_shape) == 2:  # (2,N) or (2,1)
        points = points_world
    else:  # (2,N,1) or other
        points = points_world.squeeze(-1)
    
    # Create rotation matrix (world to robot)
    cos_theta = torch.cos(torch.tensor(-theta, device=robot_state.device))
    sin_theta = torch.sin(torch.tensor(-theta, device=robot_state.device))
    R = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ], device=robot_state.device, dtype=robot_state.dtype)
    
    # Transform: p_robot = R * (p_world - p_robot_pos)
    points_translated = points - robot_pos  # Translate to robot center
    points_robot = torch.matmul(R, points_translated)  # Rotate to robot frame
    
    # Reshape back to original shape
    if len(original_shape) == 1:  # (2,)
        points_robot = points_robot.flatten()
    elif len(original_shape) == 3:  # (2,N,1)
        points_robot = points_robot.unsqueeze(-1)
    
    return points_robot

def transform_points_to_world_frame(points_robot, robot_state):
    """
    Transform points from robot frame to world frame.
    
    Args:
        points_robot: Points in robot frame (torch.Tensor)
            Shape can be (2,), (2,1), (2,N), or (2,N,1)
        robot_state: Robot state tensor [x, y, theta] (torch.Tensor)
    
    Returns:
        points_world: Points in world frame (torch.Tensor)
            Will have same shape as points_robot
    """
    # Handle different input shapes
    original_shape = points_robot.shape
    
    # Extract robot position and orientation
    robot_pos = robot_state[:2].reshape(2, 1)
    theta = robot_state[2].item()
    
    # Reshape points_robot to (2,N) for easier processing
    if len(original_shape) == 1:  # (2,)
        points = points_robot.reshape(2, 1)
    elif len(original_shape) == 2:  # (2,N) or (2,1)
        points = points_robot
    else:  # (2,N,1) or other
        points = points_robot.squeeze(-1)
    
    # Create rotation matrix (robot to world)
    cos_theta = torch.cos(torch.tensor(theta, device=robot_state.device))
    sin_theta = torch.sin(torch.tensor(theta, device=robot_state.device))
    R = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ], device=robot_state.device, dtype=robot_state.dtype)
    
    # Transform: p_world = R * p_robot + p_robot_pos
    points_rotated = torch.matmul(R, points)  # Rotate from robot to world
    points_world = points_rotated + robot_pos  # Translate to robot position
    
    # Reshape back to original shape
    if len(original_shape) == 1:  # (2,)
        points_world = points_world.flatten()
    elif len(original_shape) == 3:  # (2,N,1)
        points_world = points_world.unsqueeze(-1)
    
    return points_world

def transform_to_local_frame(global_point, translation, rotation):
    """
    Transform a point from global frame to local frame using translation and rotation
    
    Args:
        global_point: Point in global frame (torch.Tensor or np.ndarray)
        translation: Translation vector (torch.Tensor or np.ndarray)
        rotation: Rotation angle in radians (float or torch.Tensor)
    
    Returns:
        local_point: Point in local frame (same type as input), always with shape (2,)
    """
    # Print debug info about input
    print(f"transform_to_local_frame input:")
    print(f"  global_point: type={type(global_point)}, shape={global_point.shape if hasattr(global_point, 'shape') else np.shape(global_point)}")
    print(f"  translation: type={type(translation)}, shape={translation.shape if hasattr(translation, 'shape') else np.shape(translation) if hasattr(translation, 'shape') else 'scalar'}")
    print(f"  rotation: type={type(rotation)}, value={rotation}")
    
    # Determine if we're working with torch tensors or numpy arrays
    is_torch = isinstance(global_point, torch.Tensor)
    
    if is_torch:
        # Handle torch tensors
        # Ensure point has shape [2]
        if global_point.ndim > 1:
            point = global_point.reshape(-1)[:2]
        else:
            point = global_point[:2]
            
        # Handle translation
        if not isinstance(translation, torch.Tensor):
            trans = torch.tensor(translation, device=point.device, dtype=point.dtype)
        else:
            trans = translation
            if trans.ndim > 1 and trans.shape[0] > 2:
                trans = trans[:2]
            if trans.ndim > 1 and trans.shape[1] > 1:
                trans = trans[:, 0]
                
        # Create rotation matrix
        if isinstance(rotation, torch.Tensor):
            rot_value = rotation.item() if rotation.numel() == 1 else rotation[0].item()
        else:
            rot_value = rotation
            
        cos_rot = torch.cos(torch.tensor(rot_value))
        sin_rot = torch.sin(torch.tensor(rot_value))
        rot_matrix = torch.tensor([[cos_rot, -sin_rot], 
                                  [sin_rot, cos_rot]], 
                                 device=point.device, dtype=point.dtype)
        
        # Apply transformation: R^T * (p - t)
        # First subtract translation
        translated = point - trans
        
        # Then apply inverse rotation (transpose of rotation matrix)
        # Ensure translated is reshaped properly for matrix multiplication
        if len(translated.shape) == 1:
            translated_reshaped = translated.reshape(-1, 1)  # Make it a column vector
            local_point = torch.matmul(rot_matrix.transpose(0, 1), translated_reshaped)
            local_point = local_point.flatten()  # Convert back to 1D tensor
        else:
            local_point = torch.matmul(rot_matrix.transpose(0, 1), translated)
            
        # Ensure we only return a 2D vector
        if local_point.shape[0] > 2:
            print(f"WARNING: local_point has {local_point.shape[0]} dimensions, truncating to first 2")
            local_point = local_point[:2]
    else:
        # Handle numpy arrays
        # Ensure point has shape [2]
        if global_point.ndim > 1:
            point = global_point.reshape(-1)[:2]
        else:
            point = global_point[:2]
            
        # Handle translation
        if isinstance(translation, torch.Tensor):
            trans = translation.detach().cpu().numpy()
            if trans.ndim > 1 and trans.shape[0] > 2:
                trans = trans[:2]
            if trans.ndim > 1 and trans.shape[1] > 1:
                trans = trans[:, 0]
        else:
            trans = np.array(translation)
            
        # Create rotation matrix
        if isinstance(rotation, torch.Tensor):
            rot_value = rotation.item() if rotation.numel() == 1 else rotation[0].item()
        else:
            rot_value = rotation
            
        cos_rot = np.cos(rot_value)
        sin_rot = np.sin(rot_value)
        rot_matrix = np.array([[cos_rot, -sin_rot], 
                              [sin_rot, cos_rot]])
        
        # Apply transformation: R^T * (p - t)
        translated = point - trans
        
        # Ensure translated is reshaped properly for matrix multiplication
        if len(translated.shape) == 1:
            translated_reshaped = translated.reshape(-1, 1)  # Make it a column vector
            local_point = np.matmul(rot_matrix.T, translated_reshaped)
            local_point = local_point.flatten()  # Convert back to 1D array
        else:
            local_point = np.matmul(rot_matrix.T, translated)
            
        # Ensure we only return a 2D vector
        if local_point.shape[0] > 2:
            print(f"WARNING: local_point has {local_point.shape[0]} dimensions, truncating to first 2")
            local_point = local_point[:2]
    
    # Print debug info about output
    print(f"transform_to_local_frame output:")
    print(f"  local_point: type={type(local_point)}, shape={local_point.shape if hasattr(local_point, 'shape') else np.shape(local_point)}")
    
    return local_point