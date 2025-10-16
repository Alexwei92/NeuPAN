import numpy as np

def print_transformation_debug_info(robot_state, robot_state_trans, translation, points=None):
    """
    Print detailed debugging information about transformations without using matplotlib.
    
    Args:
        robot_state: Original robot state [x, y, theta]
        robot_state_trans: Translated robot state
        translation: Translation vector
        points: Optional points to transform
    """
    # Convert all inputs to numpy arrays
    if hasattr(robot_state, 'detach'):
        robot_state_np = robot_state.detach().cpu().numpy().flatten()
    else:
        robot_state_np = np.array(robot_state).flatten()
    
    if hasattr(robot_state_trans, 'detach'):
        robot_state_trans_np = robot_state_trans.detach().cpu().numpy().flatten()
    else:
        robot_state_trans_np = np.array(robot_state_trans).flatten()
        
    if hasattr(translation, 'detach'):
        translation_np = translation.detach().cpu().numpy().flatten()
    else:
        translation_np = np.array(translation).flatten()
        
    # Print basic information
    print("\n=== Transformation Debug Information ===")
    print(f"Original robot state: position=({robot_state_np[0]:.4f}, {robot_state_np[1]:.4f}), theta={robot_state_np[2]:.4f}")
    print(f"Translated robot state: position=({robot_state_trans_np[0]:.4f}, {robot_state_trans_np[1]:.4f}), theta={robot_state_trans_np[2]:.4f}")
    print(f"Translation vector: ({translation_np[0]:.4f}, {translation_np[1]:.4f})")
    
    # Calculate difference between positions
    pos_diff = robot_state_trans_np[:2] - robot_state_np[:2]
    print(f"Position difference: ({pos_diff[0]:.4f}, {pos_diff[1]:.4f})")
    
    # Calculate error between expected and actual translation
    trans_error = pos_diff - translation_np
    print(f"Translation error: ({trans_error[0]:.4f}, {trans_error[1]:.4f})")
    print(f"Translation error magnitude: {np.linalg.norm(trans_error):.4f}")
    
    # If points are provided, transform a few sample points
    if points is not None:
        if hasattr(points, 'detach'):
            points_np = points.detach().cpu().numpy()
        else:
            points_np = np.array(points)
            
        # Handle different shapes
        if len(points_np.shape) == 3:  # (2, N, 1)
            points_np = points_np.squeeze(-1)
            
        # Only transform up to 3 points for brevity
        num_points = min(3, points_np.shape[1])
        
        print("\n=== Sample Point Transformations ===")
        
        for i in range(num_points):
            p_world = points_np[:2, i].reshape(2, 1)
            
            # Calculate rotation matrices
            theta = robot_state_np[2]
            theta_trans = robot_state_trans_np[2]
            
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            
            R_trans = np.array([
                [np.cos(theta_trans), -np.sin(theta_trans)],
                [np.sin(theta_trans), np.cos(theta_trans)]
            ])
            
            # Transform point to original robot frame
            p_robot = np.matmul(R.T, (p_world - robot_state_np[:2].reshape(2, 1)))
            
            # Transform point to translated robot frame
            p_robot_trans = np.matmul(R_trans.T, (p_world - robot_state_trans_np[:2].reshape(2, 1)))
            
            print(f"\nPoint {i} in world frame: ({p_world[0, 0]:.4f}, {p_world[1, 0]:.4f})")
            print(f"Point {i} in original robot frame: ({p_robot[0, 0]:.4f}, {p_robot[1, 0]:.4f})")
            print(f"Point {i} in translated robot frame: ({p_robot_trans[0, 0]:.4f}, {p_robot_trans[1, 0]:.4f})")
            print(f"Difference between frames: ({(p_robot_trans - p_robot)[0, 0]:.4f}, {(p_robot_trans - p_robot)[1, 0]:.4f})")
            
    print("\n=== End of Transformation Debug ===")