#!/usr/bin/env python3
"""
This script analyzes the differences between:
1. Direct DUNE neural network evaluation (similar to dune_evaluation.py)
2. DUNE evaluation in mosaic_dune_eval.py with real-world points

The goal is to understand why the two approaches yield different results
despite using the same model and transformation logic.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import argparse
import os
import sys
import pickle
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from neupan.blocks.dune_train import DUNETrain
from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor, tensor_to_np, to_device
from neupan import neupan
import irsim
from compute_groundtruth import compute_ground_truth

def load_model_and_robot_info(model_path):
    """Load the DUNE model and corresponding robot information."""
    # Attempt to load the train_dict.pkl file in the same directory
    model_dir = os.path.dirname(model_path)
    train_dict_path = os.path.join(model_dir, "train_dict.pkl")
    
    try:
        with open(train_dict_path, "rb") as f:
            train_dict = pickle.load(f)
            print(f"Loaded training parameters: {train_dict.keys()}")
            
            # Extract the relevant information
            robot_G = train_dict.get("robot_G")
            robot_h = train_dict.get("robot_h")
            model_template = train_dict.get("model")
            data_range = train_dict.get("data_range", [-25, -25, 25, 25])
            
            # Check that we have all the required information
            if robot_G is None or robot_h is None or model_template is None:
                raise ValueError("Missing required information in train_dict.pkl")
            
            # Create a new model with the same architecture
            model = model_template
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
            return model, robot_G, robot_h, data_range
            
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading train_dict.pkl: {e}")
        print("Trying to load model directly...")
        
        # If we can't load the train_dict, we'll try to load just the model
        # and get the robot info from the planner
        model = ObsPointNet(2, edge_dim=8)  # Default size, might need to be adjusted
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, None, None, [-25, -25, 25, 25]

def generate_grid_points(data_range, num_points=100):
    """Generate a grid of points within the given range."""
    x = np.linspace(data_range[0], data_range[2], int(np.sqrt(num_points)))
    y = np.linspace(data_range[1], data_range[3], int(np.sqrt(num_points)))
    
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.flatten(), yy.flatten()])
    
    return points

def get_simulation_points(env_file, planner_file):
    """Get points from the simulation environment."""
    env = irsim.make(env_file)
    env.step()
    neupan_planner = neupan.init_from_yaml(planner_file, device='cpu', time_print=False)

    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()

    points = neupan_planner.scan_to_point(robot_state, lidar_scan)
    obs_points = np_to_tensor(points) if points is not None else None
    
    # Get the robot G and h matrices from the planner
    robot_G = neupan_planner.robot.G
    robot_h = neupan_planner.robot.h
    
    return obs_points, robot_state, robot_G, robot_h, neupan_planner

def transform_to_robot_frame(points, robot_state):
    """Transform points from world frame to robot frame."""
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = np.asarray(points)
        
    if isinstance(robot_state, torch.Tensor):
        state = robot_state.reshape((3, 1)).detach().cpu().numpy()
    else:
        state = np.asarray(robot_state).reshape((3, 1))
    
    # Extract translation and rotation
    trans = state[0:2]
    theta = state[2, 0]
    
    # Create rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    
    # Apply transformation - subtract translation, then rotate
    points_centered = points_np[:2] - trans
    points_robot_frame = R.T @ points_centered
    
    return points_robot_frame

def compare_neural_vs_solver(model, robot_G, robot_h, points, robot_state=None):
    """Compare neural network outputs to solver outputs."""
    
    # Setup device
    device = next(model.parameters()).device
    
    # If robot_state is provided, transform points to robot frame
    if robot_state is not None:
        points_robot_frame = transform_to_robot_frame(points, robot_state)
    else:
        # Points are already in robot frame
        if isinstance(points, torch.Tensor):
            points_robot_frame = points.detach().cpu().numpy()
        else:
            points_robot_frame = points
    
    # Convert to torch tensor for model input
    points_tensor = torch.tensor(points_robot_frame, dtype=torch.float32).to(device)
    
    # Make sure points are in the right shape (2, N)
    if points_tensor.shape[0] != 2 and points_tensor.shape[1] == 2:
        points_tensor = points_tensor.T
    
    # Run through the neural network
    with torch.no_grad():
        # DUNE expects input as (N, 2)
        mu_nn = model(points_tensor.T).T
    
    # Compute distance using the network output
    G_tensor = robot_G.to(device)
    h_tensor = robot_h.to(device)
    
    # Create a rotation matrix (identity for simple comparison)
    R = torch.eye(2, device=device)
    
    # Compute lambda and distance using the network's mu
    lam_nn = -R @ G_tensor.T @ mu_nn
    
    # Calculate distance for each point
    distances_nn = []
    for i in range(points_tensor.shape[1]):
        p = points_tensor[:, i:i+1]
        obj = G_tensor @ p - h_tensor
        dist = (mu_nn[:, i:i+1].T @ obj).item()
        distances_nn.append(dist)
    
    # Now use the solver for ground truth
    if robot_state is None:
        # For direct evaluation without a robot state
        robot_state_zero = torch.zeros(3, dtype=torch.float32)
    else:
        robot_state_zero = robot_state
    
    # Get ground truth using solver
    mu_gt, lam_gt, dist_gt, p0_list = compute_ground_truth(
        robot_G, robot_h, points[:2], robot_state=robot_state_zero
    )
    
    # Convert to numpy arrays for easier comparison
    mu_nn_np = mu_nn.detach().cpu().numpy()
    lam_nn_np = lam_nn.detach().cpu().numpy()
    
    # Create result arrays for comparison
    mu_diffs = []
    lam_diffs = []
    dist_diffs = []
    
    # Compare each point's results
    for i in range(min(len(mu_gt), mu_nn_np.shape[1])):
        mu_gt_i = mu_gt[i].detach().cpu().numpy()
        lam_gt_i = lam_gt[i].detach().cpu().numpy()
        
        # Normalize for comparison
        mu_nn_i = mu_nn_np[:, i:i+1]
        lam_nn_i = lam_nn_np[:, i:i+1]
        
        # Calculate cosine similarity for direction comparison
        mu_cos_sim = np.dot(mu_gt_i.flatten(), mu_nn_i.flatten()) / (
            np.linalg.norm(mu_gt_i) * np.linalg.norm(mu_nn_i) + 1e-10
        )
        
        lam_cos_sim = np.dot(lam_gt_i.flatten(), lam_nn_i.flatten()) / (
            np.linalg.norm(lam_gt_i) * np.linalg.norm(lam_nn_i) + 1e-10
        )
        
        # Calculate relative differences in distances
        if abs(dist_gt[i]) > 1e-10:
            dist_rel_diff = abs(distances_nn[i] - dist_gt[i]) / abs(dist_gt[i])
        else:
            dist_rel_diff = abs(distances_nn[i] - dist_gt[i])
        
        mu_diffs.append(1.0 - mu_cos_sim)  # Difference from perfect alignment
        lam_diffs.append(1.0 - lam_cos_sim)
        dist_diffs.append(dist_rel_diff)
    
    # Create summaries
    mu_diff_avg = np.mean(mu_diffs)
    lam_diff_avg = np.mean(lam_diffs)
    dist_diff_avg = np.mean(dist_diffs)
    
    # Detailed results for the first few points
    details = []
    for i in range(min(5, len(mu_gt))):
        details.append({
            "point": p0_list[i].flatten(),
            "mu_gt": mu_gt[i].detach().cpu().numpy().flatten(),
            "mu_nn": mu_nn_np[:, i].flatten(),
            "lam_gt": lam_gt[i].detach().cpu().numpy().flatten(),
            "lam_nn": lam_nn_np[:, i].flatten(),
            "dist_gt": dist_gt[i],
            "dist_nn": distances_nn[i],
            "mu_cos_sim": 1.0 - mu_diffs[i],
            "lam_cos_sim": 1.0 - lam_diffs[i],
            "dist_rel_diff": dist_diffs[i]
        })
    
    return {
        "mu_diff_avg": mu_diff_avg,
        "lam_diff_avg": lam_diff_avg,
        "dist_diff_avg": dist_diff_avg,
        "mu_diffs": mu_diffs,
        "lam_diffs": lam_diffs,
        "dist_diffs": dist_diffs,
        "details": details
    }

def analyze_point_distribution(points_world, points_robot, data_range):
    """Analyze the distribution of points relative to the training range."""
    
    # Convert to numpy if needed
    if isinstance(points_world, torch.Tensor):
        points_world = points_world.detach().cpu().numpy()
    
    if isinstance(points_robot, torch.Tensor):
        points_robot = points_robot.detach().cpu().numpy()
    
    # If points are (2,N), make them (N,2) for easier analysis
    if points_world.shape[0] == 2:
        points_world = points_world.T
    
    if points_robot.shape[0] == 2:
        points_robot = points_robot.T
    
    # Count points inside training range
    x_in_range = (points_robot[:, 0] >= data_range[0]) & (points_robot[:, 0] <= data_range[2])
    y_in_range = (points_robot[:, 1] >= data_range[1]) & (points_robot[:, 1] <= data_range[3])
    points_in_range = np.logical_and(x_in_range, y_in_range)
    
    num_in_range = np.sum(points_in_range)
    percent_in_range = num_in_range / len(points_in_range) * 100
    
    # Find min/max of actual points
    x_min, x_max = np.min(points_robot[:, 0]), np.max(points_robot[:, 0])
    y_min, y_max = np.min(points_robot[:, 1]), np.max(points_robot[:, 1])
    
    # Calculate how far points are from the training range
    x_out_min = np.min(points_robot[~x_in_range, 0]) if np.any(~x_in_range) else None
    x_out_max = np.max(points_robot[~x_in_range, 0]) if np.any(~x_in_range) else None
    y_out_min = np.min(points_robot[~y_in_range, 1]) if np.any(~y_in_range) else None
    y_out_max = np.max(points_robot[~y_in_range, 1]) if np.any(~y_in_range) else None
    
    return {
        "num_total": len(points_in_range),
        "num_in_range": num_in_range,
        "percent_in_range": percent_in_range,
        "actual_range": [x_min, y_min, x_max, y_max],
        "out_of_bounds": {
            "x_min": x_out_min,
            "x_max": x_out_max,
            "y_min": y_out_min,
            "y_max": y_out_max
        }
    }

def visualize_comparison(points_world, points_robot, training_range, results, out_path=None):
    """Visualize the comparison between neural network and solver outputs."""
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot point distributions
    ax = axs[0, 0]
    if isinstance(points_world, torch.Tensor):
        points_world = points_world.detach().cpu().numpy()
    if points_world.shape[0] == 2:
        points_world = points_world.T
    
    if isinstance(points_robot, torch.Tensor):
        points_robot = points_robot.detach().cpu().numpy()
    if points_robot.shape[0] == 2:
        points_robot = points_robot.T
    
    ax.scatter(points_world[:, 0], points_world[:, 1], s=10, label="World Frame")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Points in World Frame")
    ax.grid(True)
    ax.legend()
    
    # Points in robot frame with training range
    ax = axs[0, 1]
    ax.scatter(points_robot[:, 0], points_robot[:, 1], s=10)
    
    # Draw the training range
    rect = plt.Rectangle(
        (training_range[0], training_range[1]),
        training_range[2] - training_range[0],
        training_range[3] - training_range[1],
        linewidth=2, edgecolor='r', facecolor='none', label="Training Range"
    )
    ax.add_patch(rect)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Points in Robot Frame vs Training Range")
    ax.grid(True)
    ax.legend()
    
    # Plot histograms of differences
    ax = axs[1, 0]
    ax.hist(results["mu_diffs"], bins=30, alpha=0.5, label="mu diff")
    ax.hist(results["lam_diffs"], bins=30, alpha=0.5, label="lambda diff")
    ax.set_xlabel("Cosine Distance (0=identical, 2=opposite)")
    ax.set_ylabel("Count")
    ax.set_title("Direction Differences (Neural vs Solver)")
    ax.grid(True)
    ax.legend()
    
    ax = axs[1, 1]
    ax.hist(results["dist_diffs"], bins=30)
    ax.set_xlabel("Relative Distance Difference")
    ax.set_ylabel("Count")
    ax.set_title("Distance Differences (Neural vs Solver)")
    ax.grid(True)
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        print(f"Visualization saved to {out_path}")
    else:
        plt.show()

def main(args):
    # 1. Load the DUNE model and robot info
    model, model_G, model_h, training_range = load_model_and_robot_info(args.model_path)
    
    # 2. Get points from simulation
    obs_points, robot_state, sim_G, sim_h, neupan_planner = get_simulation_points(
        args.env_file, args.planner_file
    )
    
    # Use the G and h from the model if available, otherwise use from simulation
    robot_G = model_G if model_G is not None else sim_G
    robot_h = model_h if model_h is not None else sim_h
    
    print(f"Robot G shape: {robot_G.shape}, Robot h shape: {robot_h.shape}")
    print(f"Training range: {training_range}")
    
    # Convert robot state to tensor if needed
    if not isinstance(robot_state, torch.Tensor):
        robot_state_tensor = torch.tensor(robot_state[:3], dtype=torch.float32)
    else:
        robot_state_tensor = robot_state
    
    # 3. Transform points to robot frame for analysis
    points_robot_frame = transform_to_robot_frame(obs_points, robot_state_tensor)
    
    # 4. Analyze point distribution relative to training range
    dist_analysis = analyze_point_distribution(
        obs_points[:2].T, points_robot_frame.T, training_range
    )
    
    print("=== Point Distribution Analysis ===")
    print(f"Total points: {dist_analysis['num_total']}")
    print(f"Points in training range: {dist_analysis['num_in_range']} ({dist_analysis['percent_in_range']:.1f}%)")
    print(f"Actual range: X [{dist_analysis['actual_range'][0]:.2f}, {dist_analysis['actual_range'][2]:.2f}], "
          f"Y [{dist_analysis['actual_range'][1]:.2f}, {dist_analysis['actual_range'][3]:.2f}]")
    
    if dist_analysis['percent_in_range'] < 100:
        print("Out-of-bounds points:")
        ob = dist_analysis['out_of_bounds']
        if ob['x_min'] is not None:
            print(f"  X min: {ob['x_min']:.2f} (training range: {training_range[0]:.2f})")
        if ob['x_max'] is not None:
            print(f"  X max: {ob['x_max']:.2f} (training range: {training_range[2]:.2f})")
        if ob['y_min'] is not None:
            print(f"  Y min: {ob['y_min']:.2f} (training range: {training_range[1]:.2f})")
        if ob['y_max'] is not None:
            print(f"  Y max: {ob['y_max']:.2f} (training range: {training_range[3]:.2f})")
    
    # 5. Compare neural network vs solver for simulation points
    print("\n=== Comparing Neural Network vs. Solver for Simulation Points ===")
    sim_results = compare_neural_vs_solver(model, robot_G, robot_h, obs_points, robot_state_tensor)
    
    print(f"Average mu difference: {sim_results['mu_diff_avg']:.4f}")
    print(f"Average lambda difference: {sim_results['lam_diff_avg']:.4f}")
    print(f"Average distance difference: {sim_results['dist_diff_avg']:.4f}")
    
    print("\nDetailed comparison for first few points:")
    for i, detail in enumerate(sim_results['details']):
        print(f"\n--- Point {i} ---")
        print(f"Point (robot frame): {detail['point']}")
        print(f"Mu cosine similarity: {detail['mu_cos_sim']:.4f}")
        print(f"Lambda cosine similarity: {detail['lam_cos_sim']:.4f}")
        print(f"Distance relative diff: {detail['dist_rel_diff']:.4f}")
        
        if args.verbose:
            print(f"Mu GT: {detail['mu_gt']}")
            print(f"Mu NN: {detail['mu_nn']}")
            print(f"Lambda GT: {detail['lam_gt']}")
            print(f"Lambda NN: {detail['lam_nn']}")
            print(f"Distance GT: {detail['dist_gt']:.4f}")
            print(f"Distance NN: {detail['dist_nn']:.4f}")
    
    # 6. Generate and test grid points inside the training range
    if args.test_grid:
        print("\n=== Testing Grid Points Inside Training Range ===")
        grid_points = generate_grid_points(training_range, args.num_points)
        grid_results = compare_neural_vs_solver(model, robot_G, robot_h, grid_points)
        
        print(f"Grid Points - Average mu difference: {grid_results['mu_diff_avg']:.4f}")
        print(f"Grid Points - Average lambda difference: {grid_results['lam_diff_avg']:.4f}")
        print(f"Grid Points - Average distance difference: {grid_results['dist_diff_avg']:.4f}")
    else:
        grid_results = None
    
    # 7. Visualize the results
    if args.visualize:
        out_path = args.output if args.output else None
        visualize_comparison(
            obs_points[:2], points_robot_frame, 
            training_range, sim_results, out_path
        )
    
    return {
        "simulation_results": sim_results,
        "grid_results": grid_results,
        "distribution_analysis": dist_analysis
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DUNE neural network vs solver differences")
    parser.add_argument("--model-path", type=str, 
                        default="../dune_train/model/acker_robot_square_scale/poly_0/model_5000.pth",
                        help="Path to the DUNE model .pth file")
    parser.add_argument("--env-file", type=str, 
                        default="env.yaml", 
                        help="Path to environment yaml file")
    parser.add_argument("--planner-file", type=str, 
                        default="planner.yaml", 
                        help="Path to planner yaml file")
    parser.add_argument("--num-points", type=int, 
                        default=100, 
                        help="Number of grid points to test")
    parser.add_argument("--test-grid", action="store_true", 
                        help="Generate and test a grid of points inside training range")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize the results")
    parser.add_argument("--output", type=str, 
                        default=None,
                        help="Output path for visualization")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print detailed comparison information")
    
    args = parser.parse_args()
    main(args)