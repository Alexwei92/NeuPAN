"""
DUNE Evaluation Script

This script directly evaluates the DUNE network against the ground truth solver
using points sampled in the same way as during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from neupan.blocks.dune_train import DUNETrain
from neupan.blocks.obs_point_net import ObsPointNet
from compute_groundtruth import compute_ground_truth
import argparse
import os

def main(args):
    # Load the model and robot specifications
    if args.model_path:
        model_path = args.model_path
    else:
        # Default path
        model_path = "../dune_train/model/acker_robot_square_scale/poly_0/model_5000.pth"
    
    print(f"Loading model from: {model_path}")
    
    # Get the robot G and h from the model path
    if "acker_robot_square_scale" in model_path:
        # Square robot (same as in results.txt)
        G = torch.tensor([
            [-1., 0.],
            [0., -1.],
            [1., 0.],
            [0., 1.],
        ], device=args.device)
        
        h = torch.tensor([
            [1.],
            [1.],
            [1.],
            [1.],
        ], device=args.device)
    else:
        # Default to a simple square if unknown model
        print("Using default square robot geometry")
        G = torch.tensor([
            [-1., 0.],
            [0., -1.],
            [1., 0.],
            [0., 1.],
        ], device=args.device)
        
        h = torch.tensor([
            [1.],
            [1.],
            [1.],
            [1.],
        ], device=args.device)
    
    # Create the neural network
    edge_dim = G.shape[0]  # Number of edges/constraints
    model = ObsPointNet(2, edge_dim)  # 2D points -> edge_dim outputs
    model.to(args.device)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    print(f"Model loaded successfully. Edge dimension: {edge_dim}")
    
    # Generate test points exactly as in training
    # Use the same range as during training: [-25, -25, 25, 25]
    if args.range:
        data_range = args.range
    else:
        data_range = [-25, -25, 25, 25]  # Default from training
    
    print(f"Sampling points in range: {data_range}")
    
    # Create test points
    num_samples = args.num_points
    print(f"Generating {num_samples} test points...")
    
    # Sample points in the same way as during training
    rand_p = np.random.uniform(
        low=data_range[:2], high=data_range[2:], size=(num_samples, 2)
    )
    
    # Create a dummy DUNETrain for the ground truth
    dummy_model = torch.nn.Linear(1, 1)
    dummy_model.to(args.device)
    dt = DUNETrain(dummy_model, G, h, "/tmp")
    
    # Lists to store results
    nn_mu_list = []
    nn_lambda_list = []
    nn_distance_list = []
    gt_mu_list = []
    gt_lambda_list = []
    gt_distance_list = []
    p_list = []
    
    # For each point, compute:
    # 1. Neural network prediction
    # 2. Ground truth from solver
    print("Computing predictions and ground truth...")
    
    # Process all points as a batch with the neural network
    all_points = torch.tensor(rand_p, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        nn_outputs = model(all_points)  # Shape: [num_samples, edge_dim]
    
    # Reshape outputs for consistency with individual processing
    nn_outputs = nn_outputs.T  # Shape: [edge_dim, num_samples]
    
    # Process each point individually for detailed comparison
    for i, p_np in enumerate(rand_p):
        # Shape the point for processing - this is how the solver expects it
        p = p_np.reshape(2, 1)
        p_list.append(p)
        
        # 1. Neural network prediction (already computed in batch above)
        nn_mu = nn_outputs[:, i:i+1]  # Extract this point's predictions
        
        # Calculate distance as in DUNE.cal_objective_distance
        p_tensor = torch.tensor(p, dtype=torch.float32, device=args.device)
        temp = (G @ p_tensor - h)
        nn_distance = float((nn_mu.T @ temp).item())
        
        # Calculate lambda as in DUNE.forward
        # Use identity matrix as R since we're in the original coordinate frame
        R = torch.eye(2, device=args.device)
        nn_lambda = -R @ G.T @ nn_mu
        
        # Save neural network predictions
        nn_mu_list.append(nn_mu)
        nn_lambda_list.append(nn_lambda)
        nn_distance_list.append(nn_distance)
        
        # 2. Ground truth from solver
        try:
            obj_value, mu_value = dt.prob_solve(p)
            
            # Convert solver outputs to tensors
            mu_t = torch.tensor(mu_value, dtype=torch.float32, device=args.device)
            
            # Ensure mu follows the constraint ||G.T @ mu|| <= 1 exactly as in DUNE
            G_t_mu = G.T @ mu_t
            G_t_mu_norm = torch.norm(G_t_mu)
            if G_t_mu_norm > 1:
                # Rescale mu to satisfy the constraint exactly
                mu_t = mu_t / G_t_mu_norm
            
            # Calculate distance exactly as in DUNE.cal_objective_distance
            p_tensor = torch.tensor(p, dtype=torch.float32, device=args.device)
            objective = G @ p_tensor - h
            distance_value = float((mu_t.T @ objective).item())
            
            # Calculate lambda as in DUNE.forward
            lam_t = -R @ G.T @ mu_t
            
            # Save ground truth values
            gt_mu_list.append(mu_t)
            gt_lambda_list.append(lam_t)
            gt_distance_list.append(distance_value)
            
        except Exception as e:
            print(f"Error solving point {i}, p={p}: {e}")
            # Skip this point
            gt_mu_list.append(None)
            gt_lambda_list.append(None)
            gt_distance_list.append(None)
        
        # Print progress
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{num_samples} points")
    
    # Calculate errors
    mu_errors = []
    lambda_errors = []
    distance_errors = []
    mu_cosine_similarities = []
    lambda_cosine_similarities = []
    
    for i in range(num_samples):
        if gt_mu_list[i] is not None:
            # MSE error
            mu_error = torch.norm(nn_mu_list[i] - gt_mu_list[i]).item()
            lambda_error = torch.norm(nn_lambda_list[i] - gt_lambda_list[i]).item()
            distance_error = abs(nn_distance_list[i] - gt_distance_list[i])
            
            mu_errors.append(mu_error)
            lambda_errors.append(lambda_error)
            distance_errors.append(distance_error)
            
            # Cosine similarity (direction alignment)
            nn_mu_flat = nn_mu_list[i].flatten()
            gt_mu_flat = gt_mu_list[i].flatten()
            nn_lambda_flat = nn_lambda_list[i].flatten()
            gt_lambda_flat = gt_lambda_list[i].flatten()
            
            # Ensure non-zero vectors for cosine similarity
            if torch.norm(nn_mu_flat) > 1e-6 and torch.norm(gt_mu_flat) > 1e-6:
                mu_cos_sim = torch.nn.functional.cosine_similarity(
                    nn_mu_flat.unsqueeze(0), gt_mu_flat.unsqueeze(0)
                ).item()
                mu_cosine_similarities.append(mu_cos_sim)
            
            if torch.norm(nn_lambda_flat) > 1e-6 and torch.norm(gt_lambda_flat) > 1e-6:
                lambda_cos_sim = torch.nn.functional.cosine_similarity(
                    nn_lambda_flat.unsqueeze(0), gt_lambda_flat.unsqueeze(0)
                ).item()
                lambda_cosine_similarities.append(lambda_cos_sim)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean μ error: {np.mean(mu_errors):.6f}")
    print(f"Mean λ error: {np.mean(lambda_errors):.6f}")
    print(f"Mean distance error: {np.mean(distance_errors):.6f}")
    print(f"Mean μ cosine similarity: {np.mean(mu_cosine_similarities):.6f}")
    print(f"Mean λ cosine similarity: {np.mean(lambda_cosine_similarities):.6f}")
    
    # Print detailed statistics for the first few points
    print("\nDetailed comparison for first 5 points:")
    for i in range(min(5, num_samples)):
        if gt_mu_list[i] is None:
            continue
            
        p = p_list[i]
        print(f"\nPoint {i}: {p.flatten()}")
        print(f"NN μ: {nn_mu_list[i].cpu().numpy().flatten()}")
        print(f"GT μ: {gt_mu_list[i].cpu().numpy().flatten()}")
        print(f"μ error: {torch.norm(nn_mu_list[i] - gt_mu_list[i]).item():.6f}")
        
        print(f"NN λ: {nn_lambda_list[i].cpu().numpy().flatten()}")
        print(f"GT λ: {gt_lambda_list[i].cpu().numpy().flatten()}")
        print(f"λ error: {torch.norm(nn_lambda_list[i] - gt_lambda_list[i]).item():.6f}")
        
        print(f"NN distance: {nn_distance_list[i]:.6f}")
        print(f"GT distance: {gt_distance_list[i]:.6f}")
        print(f"Distance error: {abs(nn_distance_list[i] - gt_distance_list[i]):.6f}")
    
    # Create visualization
    if args.visualize:
        print("\nCreating visualizations...")
        
        # Convert lists to numpy arrays for plotting
        mu_errors_np = np.array(mu_errors)
        lambda_errors_np = np.array(lambda_errors)
        distance_errors_np = np.array(distance_errors)
        mu_cosine_similarities_np = np.array(mu_cosine_similarities)
        lambda_cosine_similarities_np = np.array(lambda_cosine_similarities)
        
        # Create directory for plots if it doesn't exist
        os.makedirs("./evaluation_plots", exist_ok=True)
        
        # Plot histograms of errors
        plt.figure(figsize=(10, 6))
        plt.hist(mu_errors_np, bins=50, alpha=0.7)
        plt.title("Histogram of μ Errors")
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.savefig("./evaluation_plots/mu_errors_histogram.png")
        
        plt.figure(figsize=(10, 6))
        plt.hist(lambda_errors_np, bins=50, alpha=0.7)
        plt.title("Histogram of λ Errors")
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.savefig("./evaluation_plots/lambda_errors_histogram.png")
        
        plt.figure(figsize=(10, 6))
        plt.hist(distance_errors_np, bins=50, alpha=0.7)
        plt.title("Histogram of Distance Errors")
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.savefig("./evaluation_plots/distance_errors_histogram.png")
        
        # Plot histograms of cosine similarities
        plt.figure(figsize=(10, 6))
        plt.hist(mu_cosine_similarities_np, bins=50, alpha=0.7)
        plt.title("Histogram of μ Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.savefig("./evaluation_plots/mu_cosine_similarities_histogram.png")
        
        plt.figure(figsize=(10, 6))
        plt.hist(lambda_cosine_similarities_np, bins=50, alpha=0.7)
        plt.title("Histogram of λ Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.savefig("./evaluation_plots/lambda_cosine_similarities_histogram.png")
        
        # Scatter plot of points colored by error
        points_np = np.array([p.flatten() for p in p_list])
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(points_np[:, 0], points_np[:, 1], c=mu_errors_np, 
                             cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='μ Error')
        plt.title("Points Colored by μ Error")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("./evaluation_plots/points_mu_error.png")
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(points_np[:, 0], points_np[:, 1], c=distance_errors_np, 
                             cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Distance Error')
        plt.title("Points Colored by Distance Error")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("./evaluation_plots/points_distance_error.png")
        
        print("Visualizations saved to ./evaluation_plots/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DUNE network against ground truth solver")
    parser.add_argument("--model-path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run evaluation on (cuda or cpu)")
    parser.add_argument("--num-points", type=int, default=1000, 
                        help="Number of points to sample for evaluation")
    parser.add_argument("--range", type=list, default=[-25, -25, 25, 25], 
                        help="Range to sample points from [x_min, y_min, x_max, y_max]")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate visualizations of the results")
    
    args = parser.parse_args()
    main(args)