import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def visualize_transformations(robot_state_tensor, robot_state_tensor_trans, 
                              translation, points_world, G, h, G_scaled=None, h_scaled=None):
    """
    Visualize the transformation between robot frames to help debug mosaic transforms.
    
    Args:
        robot_state_tensor: Original robot state [x, y, theta]
        robot_state_tensor_trans: Translated robot state for second polygon
        translation: Translation vector between polygons
        points_world: Points in world frame to visualize in both robot frames
        G, h: Original polygon constraint matrices
        G_scaled, h_scaled: Scaled polygon constraint matrices (optional)
    """
    # Print debug info about input data types
    print(f"\nInput data types for visualization:")
    print(f"robot_state_tensor: {type(robot_state_tensor)}, dtype: {robot_state_tensor.dtype if hasattr(robot_state_tensor, 'dtype') else 'n/a'}")
    print(f"robot_state_tensor_trans: {type(robot_state_tensor_trans)}, dtype: {robot_state_tensor_trans.dtype if hasattr(robot_state_tensor_trans, 'dtype') else 'n/a'}")
    print(f"translation: {type(translation)}, dtype: {translation.dtype if hasattr(translation, 'dtype') else 'n/a'}")
    print(f"points_world: {type(points_world)}, dtype: {points_world.dtype if hasattr(points_world, 'dtype') else 'n/a'}")
    
    # Handle device for torch tensors or use None for numpy arrays
    device = None
    if hasattr(robot_state_tensor, 'device'):
        device = robot_state_tensor.device
        print(f"Using device: {device}")
    
    # Convert tensors to numpy
    robot_state = robot_state_tensor.detach().cpu().numpy().flatten() if hasattr(robot_state_tensor, 'detach') else robot_state_tensor.flatten()
    robot_state_trans = robot_state_tensor_trans.detach().cpu().numpy().flatten() if hasattr(robot_state_tensor_trans, 'detach') else robot_state_tensor_trans.flatten()
    
    # Handle translation tensor which might have different shapes
    if hasattr(translation, 'detach'):
        trans_vec = translation.detach().cpu().numpy()
    else:
        trans_vec = translation
    
    # Ensure translation is a 1D array
    if len(trans_vec.shape) > 1:
        trans_vec = trans_vec.flatten()
        
    # Get rotation matrices
    theta = float(robot_state[2])
    theta_trans = float(robot_state_trans[2])
    
    # Explicitly use float64 for consistent numpy types
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)
    
    R_trans = np.array([
        [np.cos(theta_trans), -np.sin(theta_trans)],
        [np.sin(theta_trans), np.cos(theta_trans)]
    ], dtype=np.float64)
    
    # Transform points to robot frames
    if hasattr(points_world, 'detach'):
        points_world_np = points_world.detach().cpu().numpy()
    else:
        points_world_np = points_world
    
    # Handle different possible shapes of the points array
    if len(points_world_np.shape) == 3:  # (2, N, 1) shape
        points_world_np = points_world_np.squeeze(-1)
    
    # Convert robot state to numpy with consistent type
    robot_state_np = robot_state.flatten() if hasattr(robot_state, 'flatten') else np.array(robot_state, dtype=np.float64)
    robot_state_trans_np = robot_state_trans.flatten() if hasattr(robot_state_trans, 'flatten') else np.array(robot_state_trans, dtype=np.float64)
    
    # Ensure consistent numpy dtype
    points_world_np = points_world_np.astype(np.float64)
    
    points_robot = []
    points_robot_trans = []
    
    try:
        # Debug info
        print(f"points_world_np shape: {points_world_np.shape}, dtype: {points_world_np.dtype}")
        print(f"robot_state_np shape: {robot_state_np.shape}, dtype: {robot_state_np.dtype}")
        print(f"R shape: {R.shape}, dtype: {R.dtype}")
        
        # For each point in world frame, transform to both robot frames
        for i in range(points_world_np.shape[1]):
            p_world = points_world_np[:2, i].reshape(2, 1)
            
            # Get robot positions with consistent shape and type
            robot_pos = robot_state_np[:2].reshape(2, 1)
            robot_pos_trans = robot_state_trans_np[:2].reshape(2, 1)
            
            # Transform to original robot frame: p_r = R^T @ (p_w - t)
            p_robot = np.matmul(R.T, (p_world - robot_pos))
            points_robot.append(p_robot.flatten())
            
            # Transform to translated robot frame
            p_robot_trans = np.matmul(R_trans.T, (p_world - robot_pos_trans))
            points_robot_trans.append(p_robot_trans.flatten())
            
    except Exception as e:
        print(f"Error transforming points: {e}")
        import traceback
        traceback.print_exc()
        # Create dummy points if transformation fails
        if len(points_robot) == 0:
            points_robot.append(np.array([0.0, 0.0]))
        if len(points_robot_trans) == 0:
            points_robot_trans.append(np.array([0.0, 0.0]))
    
    # Convert to numpy arrays (with error handling)
    if len(points_robot) > 0:
        points_robot = np.array(points_robot)
    else:
        points_robot = np.array([[0, 0]])  # Default point if none were transformed
    
    if len(points_robot_trans) > 0:
        points_robot_trans = np.array(points_robot_trans)
    else:
        points_robot_trans = np.array([[0, 0]])  # Default point if none were transformed
    
    # Get robot polygon vertices from G and h
    if G is not None and h is not None:
        polygon_verts = get_polygon_vertices(G, h)
        if G_scaled is not None and h_scaled is not None:
            polygon_verts_scaled = get_polygon_vertices(G_scaled, h_scaled)
        else:
            polygon_verts_scaled = None
    else:
        polygon_verts = None
        polygon_verts_scaled = None
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot world frame
    ax1.set_title('World Frame')
    ax1.set_aspect('equal')
    
    # Plot robot positions in world frame
    ax1.scatter(float(robot_state[0]), float(robot_state[1]), color='blue', s=100, marker='o', label='Robot 1 Pos')
    ax1.scatter(float(robot_state_trans[0]), float(robot_state_trans[1]), color='red', s=100, marker='o', label='Robot 2 Pos')
    
    # Draw robot orientation arrows
    arrow_len = 0.5
    ax1.arrow(float(robot_state[0]), float(robot_state[1]), 
              float(arrow_len * np.cos(theta)), float(arrow_len * np.sin(theta)),
              head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    ax1.arrow(float(robot_state_trans[0]), float(robot_state_trans[1]), 
              float(arrow_len * np.cos(theta_trans)), float(arrow_len * np.sin(theta_trans)),
              head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    # Draw translation vector
    ax1.arrow(float(robot_state[0]), float(robot_state[1]), 
              float(trans_vec[0]), float(trans_vec[1]),
              head_width=0.1, head_length=0.1, fc='green', ec='green', linestyle='--',
              label='Translation')
    
    # Plot points in world frame
    if points_world_np.shape[1] > 0:
        ax1.scatter(points_world_np[0], points_world_np[1], color='black', s=50, marker='x', label='Points')
    
    # Draw robot polygons in world frame if available
    if polygon_verts is not None:
        try:
            # Transform polygon verts to world frame for robot 1
            poly1_world = []
            for p in polygon_verts:
                p_world = R @ p.reshape(2, 1) + robot_state[:2].reshape(2, 1)
                poly1_world.append(p_world.flatten())
            
            poly1 = Polygon(poly1_world, fill=False, edgecolor='blue', alpha=0.7)
            ax1.add_patch(poly1)
            
            # Transform polygon verts to world frame for robot 2
            if polygon_verts_scaled is not None:
                poly2_world = []
                for p in polygon_verts_scaled:
                    p_world = R_trans @ p.reshape(2, 1) + robot_state_trans[:2].reshape(2, 1)
                    poly2_world.append(p_world.flatten())
                
                poly2 = Polygon(poly2_world, fill=False, edgecolor='red', alpha=0.7)
                ax1.add_patch(poly2)
        except Exception as e:
            print(f"Error drawing polygons in world frame: {e}")
    
    # Plot robot frames
    ax2.set_title('Robot Frames Comparison')
    ax2.set_aspect('equal')
    
    # Plot original robot frame points
    if len(points_robot) > 0:
        ax2.scatter(points_robot[:, 0], points_robot[:, 1], color='blue', s=50, marker='o', label='Points in Robot 1 Frame')
    
    # Plot translated robot frame points
    if len(points_robot_trans) > 0:
        ax2.scatter(points_robot_trans[:, 0], points_robot_trans[:, 1], color='red', s=50, marker='x', label='Points in Robot 2 Frame')
    
    # Plot robot polygons in robot frames
    if polygon_verts is not None:
        try:
            poly1 = Polygon(polygon_verts, fill=False, edgecolor='blue', alpha=0.7)
            ax2.add_patch(poly1)
            
            if polygon_verts_scaled is not None:
                poly2 = Polygon(polygon_verts_scaled, fill=False, edgecolor='red', alpha=0.7)
                ax2.add_patch(poly2)
        except Exception as e:
            print(f"Error drawing polygons in robot frame: {e}")
    
    # Add legends and grid
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    try:
        plt.savefig('transformation_debug.png')
        print(f"Transformation visualization saved to transformation_debug.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
        # Try with a different backend
        try:
            plt.switch_backend('Agg')
            plt.savefig('transformation_debug.png')
            print(f"Transformation visualization saved to transformation_debug.png (using Agg backend)")
        except Exception as e2:
            print(f"Failed to save plot with Agg backend: {e2}")
    
    return points_robot, points_robot_trans

def get_polygon_vertices(G, h):
    """
    Extract polygon vertices from G and h matrices.
    This is an approximation method and may not work for all polygon types.
    """
    try:
        from scipy.spatial import ConvexHull
        
        # Convert inputs to numpy if they aren't already
        G_np = G
        h_np = h
        if hasattr(G, 'detach'):
            G_np = G.detach().cpu().numpy()
        if hasattr(h, 'detach'):
            h_np = h.detach().cpu().numpy()
        
        # Try to extract vertices from G and h
        # Generate a grid of points
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        points = np.vstack((X.flatten(), Y.flatten())).T
        
        # Filter points that satisfy G @ p <= h
        valid_points = []
        for p in points:
            # Handle different shapes properly
            constraints = np.dot(G_np, p.reshape(-1, 1)).flatten()
            if np.all(constraints <= h_np.flatten()):
                valid_points.append(p)
        
        if len(valid_points) > 3:
            # Compute the convex hull
            hull = ConvexHull(valid_points)
            vertices = np.array([valid_points[i] for i in hull.vertices])
            return vertices
        else:
            print("Warning: Not enough valid points for polygon, using default square")
            # Fallback to a simple square if we can't determine the vertices
            return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    except Exception as e:
        print(f"Could not extract polygon vertices: {e}")
        # Fallback to a simple square
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])