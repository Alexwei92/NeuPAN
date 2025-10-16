from neupan import neupan
import irsim
import numpy as np
import argparse
import torch
from pathlib import Path

from neupan.configuration import np_to_tensor, tensor_to_np
from neupan.util import downsample_decimation, time_it
import time
from neupan.blocks import dune_train
from compute_groundtruth import compute_ground_truth
from transform_util import transform_to_local_frame
# import warnings

# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
#     message=r"Converting [GA] to a CSC matrix; may take a while\.",
#     module=r"^ecos(\.ecos)?$",
# )


class Timer:
    @time_it("Obstacle Processing")
    def process_obstacles(self, obs_points, pan, nom_s, point_velocities):
        if obs_points is not None and not pan.no_obs:
            point_flow_list, R_list, obs_points_list = pan.generate_point_flow(
                nom_s, obs_points, point_velocities
            )

            if pan.is_multipolygon:
                mu_list, lam_list, sort_point_list = [], [], []
                for i, dune_layer in enumerate(pan.dune_layer_list):
                    mu_list_i, lam_list_i, sort_point_list_i, distance_list_i = dune_layer(
                        point_flow_list, R_list, obs_points_list
                    )
                    mu_list.append(mu_list_i)
                    lam_list.append(lam_list_i)
                    sort_point_list.append(sort_point_list_i)
            else:  
                mu_list, lam_list, sort_point_list, distance_list = pan.dune_layer(
                    point_flow_list, R_list, obs_points_list
                )
        else:
            mu_list, lam_list, sort_point_list = [], [], []
        return mu_list, lam_list, sort_point_list
    
    @time_it("Obstacle Processing")
    def process_obstacles_scale(self, obs_points, pan, robot, nom_s, point_velocities):
        # for mosaic, we have one layer to estimate the scaled polygon
        mu_list_mosaic, lam_list_mosaic, sort_point_list_mosaic, distance_list_mosaic = [],[],[],[]

        point_flow_list, R_list, obs_points_list = pan.generate_point_flow(
            nom_s, obs_points, point_velocities
        )
        # in the older version, the first vertice is the unit one, calculate it first
        mu_list, lam_list, sort_point_list, distance_list = pan.dune_layer(
                        point_flow_list, R_list, obs_points_list) 
        # print(f"mu_list: {mu_list},\n lam_list: {lam_list}")
        # print(f"sort_point_list: {sort_point_list},\n distance_list: {distance_list}")
        mu_list_mosaic.append(mu_list)
        lam_list_mosaic.append(lam_list)
        sort_point_list_mosaic.append(sort_point_list)
        distance_list_mosaic.append(distance_list)

        for (ratio, trans) in zip(robot.mosaic_ratios, robot.mosaic_translations):
            # Debug prints about dims and devices
            # try:
            #     print(f"[mosaic] ratio: {ratio}, trans.shape: {tuple(trans.shape)}, trans.device: {trans.device}")
            #     print(f"[mosaic] nom_s shape: {tuple(nom_s.shape)}, dtype: {nom_s.dtype}, device: {nom_s.device}")
            # except Exception:
            #     pass
            # 1. translate the nom_s based on the translation (vector of other square center to base center)
            nom_s_trans = nom_s.clone()
            # ensure trans has shape (2,1,1) so it broadcasts over time and any extra dims
            try:
                trans_b = trans.to(device=nom_s_trans.device, dtype=nom_s_trans.dtype).reshape(2, 1, 1)
            except Exception:
                # fallback: create a tensor from values
                trans_b = torch.tensor(trans, device=nom_s_trans.device, dtype=nom_s_trans.dtype).reshape(2, 1, 1)
            nom_s_trans[:2, ...] += trans_b
            # try:
            #     print(f"[mosaic] nom_s_trans shape: {trans}")
            # except Exception:
            #     pass
            # 2. scale only the first two rows (x, y)
            # scaled_nom_s = torch.cat((nom_s_trans[:2, :] / ratio, nom_s_trans[2:, :]), dim=0)
            # print(f"**********ratio: {ratio}**********")
            # print(f"nom_s_trans: {nom_s_trans}")
            scaled_nom_s = nom_s_trans.clone()
            # handle variable trailing dimensions (time, batch, etc.)
            scaled_nom_s[:2, ...] /= ratio
            # print(f"scaled_nom_s: {scaled_nom_s}")
            # try:
            #     print(f"[mosaic] scaled_nom_s shape: {tuple(scaled_nom_s.shape)}")
            # except Exception:
            #     pass
            # Scale obstacle points consistently in x, y
            scaled_obs = obs_points.clone()
            # print(f"scaled_obs: {scaled_obs}")
            # handle both (3,N) and (3,N,1) shapes
            scaled_obs[:2, ...] /= ratio
            # print(f"scaled_obs: {scaled_obs}")

            # try:
            #     print(f"[mosaic] scaled_obs shape: {tuple(scaled_obs.shape)}")
            # except Exception:
            #     pass
            point_velocities_scaled = point_velocities/ratio if point_velocities is not None else None
            # try:
            #     if point_velocities_scaled is not None:
            #         print(f"[mosaic] point_velocities_scaled shape: {tuple(point_velocities_scaled.shape)}")
            # except Exception:
            #     pass
            # 3. generate the point flow based on the scaled nom_s and scaled obs points
            point_flow_scaled, R_list, obs_points_scaled = pan.generate_point_flow(
                scaled_nom_s, scaled_obs, point_velocities_scaled
            )
            # 4. forward dune based on the scaled point flow and obs points
            mu_list_scaled, lam_list_scaled, sort_point_list_scaled, distance_list_scaled = pan.dune_layer(
                point_flow_scaled, R_list, obs_points_scaled)
            distance_scaled_from_unit = [d * ratio for d in distance_list_scaled]
            sort_point_scaled_from_unit = [p * ratio for p in sort_point_list_scaled]
            # print(f"mu_list_scaled:{mu_list_scaled}")
            # print(f"lam_list_scaled:{lam_list_scaled}")
            mu_list_mosaic.append(mu_list_scaled)
            lam_list_mosaic.append(lam_list_scaled)
            # Correct: append scaled points to point list, scaled distances to distance list
            sort_point_list_mosaic.append(sort_point_scaled_from_unit)
            distance_list_mosaic.append(distance_scaled_from_unit)
            # print(f"mu_scaled_from_unit:{mu_list_scaled[0]}")
            # print(f"lam_scaled_from_unit:{lam_list_scaled[0]}")
            # print(f"distance_scaled_from_unit:{distance_list_mosaic[0]}")
            # print(f"sort_point_scaled_from_unit:{sort_point_list_mosaic[0]}")
            # print(f"")
        # Keep poly-major structure: [poly][time] tensors
        mu_list = mu_list_mosaic
        lam_list = lam_list_mosaic
        sort_point_list = sort_point_list_mosaic
        distance_list = distance_list_mosaic

       # mu_scaled_from_unit = [m * 2. for m in mu_scaled_from_unit]
        return (mu_list, lam_list, sort_point_list, distance_list,)

def main(env_path, planner_path):
    """Load env and planner config and run the DUNE evaluation."""
    # Import necessary libraries
    import torch
    import numpy as np
    
    # Load env and planner config
    env_file = Path(env_path).resolve()
    planner_file = Path(planner_path).resolve()
    
    env = irsim.make(env_file)
    env.step()
    neupan_planner = neupan.init_from_yaml(planner_file, device='cuda', time_print=True)

    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()

    points = neupan_planner.scan_to_point(robot_state, lidar_scan)
    # print(f"Number of obs points: {points.shape[1] if points is not None else 0}")
    point_velocities = None

    pan = neupan_planner.pan
    robot = neupan_planner.robot
    print(neupan_planner.robot.G)
    print(neupan_planner.robot.h)
    nom_s = torch.stack([np_to_tensor(robot_state[:3]) for _ in range(pan.T+1)], dim=1)
    # points in the map frame
    obs_points = np_to_tensor(points) if points is not None else None
    print(f"Number of obs points: {obs_points.shape[1]}, type of the obs points: {type(obs_points[:,0])}")
    print(f"Number of obs points: {nom_s.shape[1]}, type of the nom_s: {type(nom_s)}")
    # I just want to choose one point form the obs_points
    # obs_points = obs_points[:,0].unsqueeze(1)
    # print(f"Number of obs points: {obs_points.shape[1]}, type of the obs points: {type(obs_points[:,0])}")
    point_velocities = None

    # Use Timer class with @time_it decorator
    timer = Timer()
    (mu_list, lam_list, sort_point_list, distance_list) = timer.process_obstacles_scale(obs_points, pan, robot, nom_s, point_velocities)
    # calculate from the ground truth
    env.draw_points(tensor_to_np(obs_points), s=20, c="r", refresh=False)

    # calculate the groundtruth mu, lambda and distance given the squares and mosaic ratios
    try:
        # obs_points is (3,N) in this script (x,y,?), keep first two rows
        obs_xy = obs_points[:2, :].clone() # Clone to avoid modifying the original
        
        # Get the current robot state to transform points to robot frame
        # This must match exactly what's passed to point_flow_list in PAN's generate_point_flow
        robot_state_tensor = torch.tensor(robot_state[:3], dtype=torch.float32)
        
        print("Robot state:", robot_state_tensor.detach().cpu().numpy())
        print("G shape:", neupan_planner.robot.G.shape, "h shape:", neupan_planner.robot.h.shape)
        print(f"Computing ground truth for {obs_xy.shape[1]} points in world frame...")
        
        # First get the point flow list from PAN to ensure we use consistent transformations
        # This must be done before compute_ground_truth to get the rotation matrix
        print("Generating PAN point flow for exact comparison...")
        point_flow_list, R_list, obs_points_list = pan.generate_point_flow(
            nom_s, obs_points, point_velocities
        )
        
        # The first element in point_flow_list is what we should compare with p0_list
        pan_points_in_robot_frame = point_flow_list[0]  # time 0
        print(f"PAN transformed {pan_points_in_robot_frame.shape[1]} points to robot frame")
        
        # Show a couple of PAN-transformed points for comparison
        for i in range(min(3, pan_points_in_robot_frame.shape[1])):
            print(f"PAN point {i} in robot frame: {pan_points_in_robot_frame[:, i].detach().cpu().numpy().flatten()}")
            
        # Use PAN's exact rotation matrix
        pan_R = R_list[0]
        print(f"Using PAN's exact rotation matrix (device: {pan_R.device}) for ground truth calculation")
            
        mu_gt, lam_gt, dist_gt, p0_list = compute_ground_truth(
            neupan_planner.robot.G, 
            neupan_planner.robot.h, 
            obs_xy,
            robot_state=robot_state_tensor,
            R_override=pan_R  # Use PAN's exact rotation matrix
        )
        print(f"Ground-truth for {len(mu_gt)} obs points computed and sorted by distance.")
        
        # Debug: Check the first few transformed points
        for i in range(min(3, len(p0_list))):
            print(f"GT point {i} in robot frame: {p0_list[i].flatten()}")
    except Exception as e:
        import traceback
        print(f"Failed to compute ground truth: {e}")
        traceback.print_exc()

    # for the first one in the returned list should be the same as the one calculated from the ground truth, compare them
    try:
        # We've already generated point_flow_list above, so we can use it directly
        # point_flow_list, R_list, and obs_points_list are already defined
        
        # If we somehow don't have them, regenerate (this should not happen)
        if not 'point_flow_list' in locals():
            print("WARNING: Regenerating point flow list - this shouldn't happen!")
            point_flow_list, R_list, obs_points_list = pan.generate_point_flow(
                nom_s, obs_points, point_velocities
            )
            
            # The first element in point_flow_list is what we should compare with p0_list
            pan_points_in_robot_frame = point_flow_list[0]  # time 0
            print(f"PAN transformed {pan_points_in_robot_frame.shape[1]} points to robot frame")
            
            # Show a couple of PAN-transformed points for comparison
            for i in range(min(3, pan_points_in_robot_frame.shape[1])):
                print(f"PAN point {i} in robot frame: {pan_points_in_robot_frame[:, i].detach().cpu().numpy().flatten()}")
        else:
            pan_points_in_robot_frame = point_flow_list[0]  # time 0
        
        # pick first polygon and time index 0
        poly_id = 0
        time_idx = 0

        # Get PAN's results
        pan_mu_poly = mu_list[poly_id]
        pan_lam_poly = lam_list[poly_id]
        pan_sorted_pts_poly = sort_point_list[poly_id] 
        pan_dist_poly = distance_list[poly_id]

        pan_mu_t = pan_mu_poly[time_idx]  # (edge_dim, num_sorted)
        pan_lam_t = pan_lam_poly[time_idx]  # (state_dim, num_sorted)
        pan_pts_t = pan_sorted_pts_poly[time_idx]  # (2 or 3, num_sorted)
        pan_dist_t = pan_dist_poly[time_idx]  # distances for time_idx

        # Now compare our ground truth to PAN's results
        num_gt = len(mu_gt)
        matched = 0
        
        print(f"\nComparing {num_gt} ground truth values with neural network results")
        
        # Convert to numpy for easier comparisons
        pan_flow_np = pan_points_in_robot_frame.detach().cpu().numpy()
        
        # Check points are properly transformed - should match between PAN and ground truth
        print("\nSample points in robot frame (first 5):")
        for i in range(min(5, pan_points_in_robot_frame.shape[1])):
            p_pan = pan_flow_np[:, i].flatten()
            p_gt = p0_list[i].flatten()
            point_dist = np.linalg.norm(p_pan - p_gt)
            print(f"  Point {i}: PAN={p_pan}, GT={p_gt}, distance={point_dist:.6f}")
        
        # Directly compare mu, lambda, and distance values for the first few points
        print("\nComparing neural network predictions with ground truth:")
        
        # Compare the neural network outputs with sorted ground truth
        print("\nComparing neural network outputs with sorted ground truth:")
            
        # Compare values for each point
        for j in range(min(len(mu_gt), 5)):
            # Get sorted ground truth values
            mu_gt_j = mu_gt[j].detach().cpu().squeeze() if hasattr(mu_gt[j], 'detach') else mu_gt[j].squeeze()
            lam_gt_j = lam_gt[j].detach().cpu().squeeze() if hasattr(lam_gt[j], 'detach') else lam_gt[j].squeeze()
            dist_gt_j = dist_gt[j]
            
            # Get the PAN outputs for comparison
            pan_mu_j = pan_mu_t[:, j].detach().cpu().squeeze()
            pan_lam_j = pan_lam_t[:, j].detach().cpu().squeeze()
            pan_dist_j = pan_dist_t[j].item() if isinstance(pan_dist_t, torch.Tensor) else float(pan_dist_t[j])
                
            # Calculate cosine similarity for direction comparison
            # Convert tensors to numpy for consistent comparison
            pan_mu_np = pan_mu_j.detach().cpu().numpy()
            mu_gt_np = mu_gt_j.detach().cpu().numpy()
            pan_lam_np = pan_lam_j.detach().cpu().numpy() 
            lam_gt_np = lam_gt_j.detach().cpu().numpy()
            
            # Calculate cosine similarity (1.0 means perfectly aligned)
            # For mu
            mu_cos_sim = np.dot(pan_mu_np.flatten(), mu_gt_np.flatten()) / (
                np.linalg.norm(pan_mu_np) * np.linalg.norm(mu_gt_np) + 1e-10
            )
            
            # For lambda
            lam_cos_sim = np.dot(pan_lam_np.flatten(), lam_gt_np.flatten()) / (
                np.linalg.norm(pan_lam_np) * np.linalg.norm(lam_gt_np) + 1e-10
            )
            
            # Relative difference in distances
            dist_rel_diff = abs(pan_dist_j - dist_gt_j) / (abs(dist_gt_j) + 1e-10)
            
            # Use the same thresholds as analyze_dune_differences.py for consistency
            mu_close = mu_cos_sim > 0.95  # 0.95 cosine similarity threshold
            lam_close = lam_cos_sim > 0.95
            # Distance close threshold
            dist_close = dist_rel_diff < 0.05  # 5% relative difference threshold
            
            if mu_close and lam_close and dist_close:
                matched += 1
            
            # Print comparison details for each point
            print(f"\n--- Point {j} comparison ---")
            print(f"Point (robot frame): {pan_flow_np[:, j]}")
            print(f"GT point: {p0_list[j].flatten()}")

            print(f"\nComparison (neural network vs ground truth):")
            print(f"Mu cosine similarity: {mu_cos_sim:.4f}")
            print(f"Lambda cosine similarity: {lam_cos_sim:.4f}")
            print(f"Distance relative diff: {dist_rel_diff:.4f}")
            
            print(f"\nDetailed values:")
            print(f"GT mu: {mu_gt_j.numpy()}")
            print(f"PAN mu: {pan_mu_j.numpy()}")
            print(f"Mu norm ratio (GT/PAN): {np.linalg.norm(mu_gt_np)/np.linalg.norm(pan_mu_np):.4f}")
            
            print(f"GT lambda: {lam_gt_j.numpy()}")
            print(f"PAN lambda: {pan_lam_j.numpy()}")
            print(f"Lambda norm ratio (GT/PAN): {np.linalg.norm(lam_gt_np)/np.linalg.norm(pan_lam_np):.4f}")
            
            print(f"GT distance: {dist_gt_j:.6f}")
            print(f"PAN distance: {pan_dist_j:.6f}")
            print(f"Distance ratio (GT/PAN): {dist_gt_j/pan_dist_j:.4f}")            # Print additional info for large differences
            if mu_cos_sim < 0.95:
                print(f"NOTE: Mu vectors have low similarity! Check mu calculation.")
            
            if lam_cos_sim < 0.95:
                print(f"NOTE: Lambda vectors have low similarity! Check lambda calculation.")
                
            if dist_rel_diff > 0.05:
                print(f"NOTE: Distance differs by more than 5%! Check distance calculation.")
                
        print(f"\nMatched {matched}/{min(len(mu_gt), 5)} points (mu+lambda+dist within tolerance).")
    except Exception as e:
        import traceback
        print(f"Comparison failed: {e}")
        traceback.print_exc()
      
    # Evaluate all mosaic polygons (squares) in the vertices list
    # For each polygon, we calculate the ground truth mu, lambda, and distance
    # Then compare with neural network predictions
    try:
        print("\n--- Evaluating All Mosaic Polygons ---")
        if len(robot.mosaic_ratios) > 0 and len(mu_list) > 1:
            print(f"Found {len(robot.mosaic_ratios)} mosaic polygons to evaluate")
            
            # First, ensure we have the same test points for all polygons
            # We'll use the first few points from the observation set
            test_point_indices = list(range(min(5, obs_xy.shape[1])))
            print(f"Using {len(test_point_indices)} test points for all polygons")
            
            # For each polygon in the mosaic (including the base polygon at index 0)
            for poly_idx in range(len(mu_list)):
                if poly_idx == 0:
                    # print(f"\n--- Base Polygon (#{poly_idx}) ---")
                    # Base polygon uses original G and h, no translation or scaling
                    poly_G = robot.G.copy()
                    poly_h = robot.h.copy()
                    poly_trans = np.zeros((2, 1))  # No translation for base
                    poly_ratio = 1.0  # No scaling for base
                else:
                    # Get the ratio and translation for this polygon
                    poly_ratio = robot.mosaic_ratios[poly_idx-1]  # Offset by 1 since ratios don't include base
                    poly_trans = robot.mosaic_translations[poly_idx-1].cpu().numpy().reshape(2, 1)
                    
                    print(f"\n--- Mosaic Polygon #{poly_idx} ---")
                    print(f"Ratio: {poly_ratio}, Translation: {poly_trans.flatten()}")
                    
                    # Scale G and h for this polygon (G stays the same, h scales by ratio)
                    poly_G = robot.G.copy()  # G doesn't change
                    poly_h = robot.h.copy() * poly_ratio  # h scales by the ratio
                
                # In the mosaic case, translations are defined from base polygon center to each polygon center
                # If this is not the base polygon, adjust the robot state according to the translation
                robot_state_tensor_poly = robot_state_tensor.clone()
                
                if poly_idx > 0:
                    # We want to adjust the robot state so it's effectively at this polygon's origin
                    # Since translation is from base to this polygon, we ADD it to move robot position
                    robot_state_tensor_poly[0] += poly_trans[0, 0]  # Add x translation
                    robot_state_tensor_poly[1] += poly_trans[1, 0]  # Add y translation
                    
                    print(f"Original robot state: {robot_state_tensor.cpu().numpy()[:2]}")
                    print(f"Translated robot state: {robot_state_tensor_poly.cpu().numpy()[:2]}")
                    print(f"Translation effect: {(robot_state_tensor_poly - robot_state_tensor).cpu().numpy()[:2]}")
                
                # Compute ground truth using polygon-specific G and h
                # Transform the original points manually to match the polygon's frame
                # For scaled/translated: we move the robot state (equivalent to moving the points)
                # Then let compute_ground_truth handle the transformation as usual
                mu_gt_poly, lam_gt_poly, dist_gt_poly, p0_list_poly = compute_ground_truth(
                    poly_G, 
                    poly_h,
                    obs_xy,  # Original points (will be transformed inside the function)
                    robot_state=robot_state_tensor_poly,
                    R_override=pan_R  # Use PAN's exact rotation matrix
                )
                if poly_idx > 0:
                    print(f"\nComparing polygon #{poly_idx} results with neural network predictions:")
                    print(f"Scale factor (ratio): {poly_ratio}")
                    print(f"G matrix shape: {poly_G.shape}, h vector shape: {poly_h.shape}")
                    print(f"Sample h values - original: {robot.h[:3]}, polygon-specific: {poly_h[:3]}")
                
                # Extract neural network results for this polygon
                time_idx = 0
            
                # Get neural network results for this polygon
                pan_mu_poly = mu_list[poly_idx]
                pan_lam_poly = lam_list[poly_idx]
                pan_sorted_pts_poly = sort_point_list[poly_idx] 
                pan_dist_poly = distance_list[poly_idx]

                pan_mu_t = pan_mu_poly[time_idx]  # (edge_dim, num_sorted)
                pan_lam_t = pan_lam_poly[time_idx]  # (state_dim, num_sorted)
                pan_pts_t = pan_sorted_pts_poly[time_idx]  # (2 or 3, num_sorted)
                pan_dist_t = pan_dist_poly[time_idx]  # distances for time_idx
                
                # Debug polygon data
                if poly_idx > 0:
                    print(f"\nPolygon {poly_idx} data summary:")
                    print(f"Number of mu vectors: {len(pan_mu_poly)}")
                    print(f"Number of lambda vectors: {len(pan_lam_poly)}")
                    print(f"Number of distance values: {len(pan_dist_poly)}")
            
                # Compare results for this polygon
                matched_poly = 0
                
                # Compare values for each point
                for j in range(min(len(mu_gt_poly), 5)):
                    # Get sorted ground truth values
                    mu_gt_j = mu_gt_poly[j].detach().cpu().squeeze() if hasattr(mu_gt_poly[j], 'detach') else mu_gt_poly[j].squeeze()
                    lam_gt_j = lam_gt_poly[j].detach().cpu().squeeze() if hasattr(lam_gt_poly[j], 'detach') else lam_gt_poly[j].squeeze()
                    dist_gt_j = dist_gt_poly[j]
                    
                    # Get the PAN outputs for comparison
                    pan_mu_j = pan_mu_t[:, j].detach().cpu().squeeze()
                    pan_lam_j = pan_lam_t[:, j].detach().cpu().squeeze()
                    pan_dist_j = pan_dist_t[j].item() if isinstance(pan_dist_t, torch.Tensor) else float(pan_dist_t[j])
                        
                    # Calculate cosine similarity
                    pan_mu_np = pan_mu_j.detach().cpu().numpy()
                    mu_gt_np = mu_gt_j.detach().cpu().numpy()
                    pan_lam_np = pan_lam_j.detach().cpu().numpy() 
                    lam_gt_np = lam_gt_j.detach().cpu().numpy()
                    
                    # Calculate cosine similarity
                    mu_cos_sim = np.dot(pan_mu_np.flatten(), mu_gt_np.flatten()) / (
                        np.linalg.norm(pan_mu_np) * np.linalg.norm(mu_gt_np) + 1e-10
                    )
                    
                    lam_cos_sim = np.dot(pan_lam_np.flatten(), lam_gt_np.flatten()) / (
                        np.linalg.norm(pan_lam_np) * np.linalg.norm(lam_gt_np) + 1e-10
                    )
                    
                    # Relative difference in distances
                    dist_rel_diff = abs(pan_dist_j - dist_gt_j) / (abs(dist_gt_j) + 1e-10)
                    
                    # Use the same thresholds
                    mu_close = mu_cos_sim > 0.95
                    lam_close = lam_cos_sim > 0.95
                    dist_close = dist_rel_diff < 0.05
                    
                    if mu_close and lam_close and dist_close:
                        matched_poly += 1
                    
                    # Print comparison details
                    if poly_idx > 0:
                        print(f"\n--- Polygon {poly_idx}, Point {j} comparison ---")
                        print(f"Point in robot frame: {p0_list_poly[j].flatten()}")
                    
                        # If this is not the base polygon, also show relationship to base points
                        if poly_idx > 0 and 'p0_list' in locals() and j < len(p0_list):
                            original_point = p0_list[j].flatten()
                            poly_point = p0_list_poly[j].flatten()
                            point_diff = poly_point - original_point
                            print(f"Original point: {original_point}")
                            print(f"Point difference (poly-original): {point_diff}")
                        
                        print(f"\nComparison (neural network vs ground truth):")
                        print(f"Mu cosine similarity: {mu_cos_sim:.4f}")
                        print(f"Lambda cosine similarity: {lam_cos_sim:.4f}")
                        print(f"Distance relative diff: {dist_rel_diff:.4f}")
                        
                        print(f"\nDetailed values:")
                        print(f"GT mu: {mu_gt_j.numpy()}")
                        print(f"PAN mu: {pan_mu_j.numpy()}")
                        print(f"Mu norm ratio (GT/PAN): {np.linalg.norm(mu_gt_np)/np.linalg.norm(pan_mu_np):.4f}")
                        
                        print(f"GT lambda: {lam_gt_j.numpy()}")
                        print(f"PAN lambda: {pan_lam_j.numpy()}")
                        print(f"Lambda norm ratio (GT/PAN): {np.linalg.norm(lam_gt_np)/np.linalg.norm(pan_lam_np):.4f}")
                        
                        print(f"GT distance: {dist_gt_j:.6f}")
                        print(f"PAN distance: {pan_dist_j:.6f}")
                        print(f"Distance ratio (GT/PAN): {dist_gt_j/pan_dist_j:.4f}")
                        
                        # Print additional info for large differences
                        if mu_cos_sim < 0.95:
                            print(f"NOTE: Mu vectors have low similarity! Check mu calculation.")
                        
                        if lam_cos_sim < 0.95:
                            print(f"NOTE: Lambda vectors have low similarity! Check lambda calculation.")
                            
                        if dist_rel_diff > 0.05:
                            print(f"NOTE: Distance differs by more than 5%! Check distance calculation.")
                            
                    print(f"\nPolygon {poly_idx}: Matched {matched_poly}/{min(len(mu_gt_poly), 5)} points (mu+lambda+dist within tolerance).")
        else:
            print("No scaled mosaic polygons found for evaluation.")
    except Exception as e:
        import traceback
        print(f"Failed to evaluate mosaic polygons: {e}")
        traceback.print_exc()
    
    print("\nEvaluation complete for all mosaic polygons.")
    input("Press Enter to close...")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    env_path_file = "env.yaml"
    planner_path_file = "planner.yaml"

    main(env_path_file, planner_path_file)
