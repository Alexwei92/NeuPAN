"""
PAN is the core class for the NeuPan algorithm. It is a proximal alternating-minimization network, consisting of NRMP and DUNE, that solves the optimization problem with numerous point-level collision avoidance constraints in each step. 

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from neupan.blocks import NRMP, DUNE
from math import inf
from typing import Optional
from neupan import configuration
from neupan.configuration import to_device, tensor_to_np
from neupan.util import downsample_decimation, time_it
from neupan import configuration

def _stack_nums_or_tensors(seq, *, dtype, device, name, last_dim=None):
    """
    Accepts: tensor, list/tuple of numbers, or list/tuple of tensors (same shape).
    Returns: tensor on (device,dtype). If last_dim is given, validate/reshape.
    """
    # already a tensor
    if isinstance(seq, torch.Tensor):
        t = seq.to(device=device, dtype=dtype)
    # list/tuple
    elif isinstance(seq, (list, tuple)):
        if len(seq) > 0 and isinstance(seq[0], torch.Tensor):
            t = torch.stack([x.to(device=device, dtype=dtype) for x in seq], dim=0)
        else:
            t = torch.as_tensor(seq, dtype=dtype, device=device)
    else:
        # single number
        t = torch.as_tensor(seq, dtype=dtype, device=device)

    # optional shape check/reshape for translations
    if last_dim is not None:
        if t.dim() == 1:
            if last_dim != 1:
                if t.numel() % last_dim != 0:
                    raise ValueError(f"{name} has numel {t.numel()} not divisible by {last_dim}.")
                t = t.view(-1, last_dim)
        elif t.size(-1) != last_dim:
            raise ValueError(f"{name} must have last dim={last_dim}, got {tuple(t.shape)}")
    return t

# --- robust normalization for mosaic params (handles lists of tensors like (2,1)) ---
def _norm_ratios(seq, *, dtype, device):
    if isinstance(seq, torch.Tensor):
        r = seq.to(device=device, dtype=dtype)
    elif isinstance(seq, (list, tuple)):
        if len(seq) > 0 and isinstance(seq[0], torch.Tensor):
            r = torch.stack([x.to(device=device, dtype=dtype).reshape(-1) for x in seq], dim=0)
        else:
            r = torch.as_tensor(seq, dtype=dtype, device=device)
    else:
        r = torch.as_tensor(seq, dtype=dtype, device=device)
    return r.flatten()  # (P,)

def _norm_translations(seq, *, dtype, device):
    """
    Accepts e.g.:
      - [[dx, dy], ...]
      - [tensor([dx, dy]), ...]
      - [tensor([[dx],[dy]]), ...]  # (2,1)
      - tensor of shape (P,2), (2,P), (P,2,1)
    Returns: (P,2)
    """
    if isinstance(seq, torch.Tensor):
        t = seq.to(device=device, dtype=dtype)
    elif isinstance(seq, (list, tuple)):
        if len(seq) > 0 and isinstance(seq[0], torch.Tensor):
            # flatten each element to shape (2,)
            t = torch.stack([x.to(device=device, dtype=dtype).reshape(-1) for x in seq], dim=0)
        else:
            t = torch.as_tensor(seq, dtype=dtype, device=device)
    else:
        t = torch.as_tensor(seq, dtype=dtype, device=device)

    # squeeze trailing singletons: (P,2,1)->(P,2), (P,2,1,1)->(P,2), etc.
    while t.dim() > 1 and t.size(-1) == 1:
        t = t.squeeze(-1)

    # coerce to (P,2)
    if t.dim() == 1:
        if t.numel() % 2 != 0:
            raise ValueError(f"mosaic_translations has {t.numel()} elements; cannot reshape to pairs of 2.")
        t = t.view(-1, 2)
    elif t.dim() == 2:
        # handle (2,P) -> (P,2)
        if t.shape[0] == 2 and t.shape[1] != 2:
            t = t.t()
        if t.shape[1] != 2:
            raise ValueError(f"mosaic_translations must end with 2 coords; got {tuple(t.shape)}")
    else:
        raise ValueError(f"mosaic_translations unexpected ndim={t.dim()} after squeeze: shape={tuple(t.shape)}")

    return t  # (P,2)

class PAN(torch.nn.Module):
    """
    Args:
        receding: int, the number of steps in the receding horizon.
        step_time: float, the time step in the MPC framework.
        robot: robot, the robot instance including the robot information.
        iter_num: int, the number of iterations in the PAN algorithm.
        dune_max_num: int, the maximum number of points considered in the DUNE model.
        nrmp_max_num: int, the maximum number of points considered in the NRMP model.
        dune_checkpoint: str, the checkpoint path for the DUNE model.
        iter_threshold: float, the threshold for the iteration to judge the convergence.
        adjust_kwargs: dict, the keyword arguments for the adjust class.
        train_kwargs: dict, the keyword arguments for the train class.
    """

    def __init__(
        self,
        receding=10,
        step_time=0.1,
        robot=None,
        iter_num=2,
        dune_max_num=100,
        nrmp_max_num=10,
        dune_checkpoint=None,
        iter_threshold=0.1,
        adjust_kwargs=dict(),
        train_kwargs=dict(),
        **kwargs,
    ) -> None:
        super(PAN, self).__init__()

        self.robot = robot
        self.T = receding
        self.dt = step_time

        self.iter_num = iter_num
        self.iter_threshold = iter_threshold
        self.nrmp_layer = NRMP(
            receding,
            step_time,
            robot,
            nrmp_max_num,
            eta=adjust_kwargs.get("eta", 10.0),
            d_max=adjust_kwargs.get("d_max", 1.0),
            d_min=adjust_kwargs.get("d_min", 0.1),
            q_s=adjust_kwargs.get("q_s", 1.0),
            p_u=adjust_kwargs.get("p_u", 1.0),
            ro_obs=adjust_kwargs.get("ro_obs", 500),
            bk=adjust_kwargs.get("bk", 0.1),
            solver=adjust_kwargs.get("solver", "ECOS"),
        )
        self.no_obs = (nrmp_max_num == 0 or dune_max_num == 0)
        self.nrmp_max_num = nrmp_max_num
        self.dune_max_num = dune_max_num
        self.is_multipolygon = robot.is_multipolygon
        self.is_mosaic = robot.is_mosaic
        self.num_of_polygons = robot.num_of_polygons
        if self.is_mosaic:
            self._min_distance = inf
            self._points = None  # Store points for mosaic case

        if not self.no_obs:
            if self.is_multipolygon:
                self.dune_layer_list = []
                for i, (robot_G, robot_h) in enumerate(zip(robot.G_list, robot.h_list)):
                    self.dune_layer_list.append(DUNE(
                        receding,
                        dune_checkpoint[i] if dune_checkpoint is not None else None,
                        robot_G,
                        robot_h,
                        dune_max_num,
                        train_kwargs,
                        robot_name=robot.name,
                        part_name='poly_' + str(i)
                    ))
            else:
                self.dune_layer = DUNE(
                    receding,
                    dune_checkpoint,
                    robot.G, # for mosaic, use the G/h of the base polygon
                    robot.h,
                    dune_max_num,
                    train_kwargs,
                    robot_name=robot.name,
                )
        else:
            self.dune_layer = None

        self.current_nom_values = [
            None,
            None,
            None,
            None,
        ]  # nom_s, nom_u, nom_lam, nom_mu

        self.printed = False

    def forward(
        self, nom_s: torch.Tensor, nom_u: torch.Tensor, ref_s: torch.Tensor, ref_us: torch.Tensor, obs_points: torch.Tensor = None, point_velocities: torch.Tensor = None, actual_vel: torch.Tensor = None
    ):
        """
        input:
            - nom_s: nominal state; (3, receding+1) 
            - nom_u: nominal control; (2, receding)
            - ref_states: reference trajectory; (3, receding+1)
            - ref_us: reference speed array;  (receding,)
            - obs_points: (2, number of obs points), point cloud, global coordinate
            - velocities: (2, number of obs points), velocity of each obs point
            - actual_vel: (2, 1), actual velocity of the robot
        output:
            - opt_vel: optimal velocity tensor; (2, receding)
            - opt_state: optimal state array  (3, receding+1)

        process:
        """

        for i in range(self.iter_num):

            if obs_points is not None and not self.no_obs:
                
                if self.is_mosaic:
                    # --- broadcasted prep across polys ---
                    # ratios = torch.as_tensor(self.robot.mosaic_ratios, dtype=nom_s.dtype, device=nom_s.device)             # (P,)
                    # translations = torch.as_tensor(self.robot.mosaic_translations, dtype=nom_s.dtype, device=nom_s.device) # (P,2)
                    # --- robust normalization for mosaic params ---


                    # Use the helper:
                    # ratios = _stack_nums_or_tensors(
                    #     self.robot.mosaic_ratios, dtype=nom_s.dtype, device=nom_s.device, name="mosaic_ratios"
                    # ).flatten()                              # (P,)

                    # translations = _stack_nums_or_tensors(
                    #     self.robot.mosaic_translations, dtype=nom_s.dtype, device=nom_s.device,
                    #     name="mosaic_translations", last_dim=2
                    # )                                        # (P,2)
                    # Normalize params once
                    ratios = _norm_ratios(self.robot.mosaic_ratios, dtype=nom_s.dtype, device=nom_s.device)  # (P,)
                    translations = _norm_translations(self.robot.mosaic_translations, dtype=nom_s.dtype, device=nom_s.device)  # (P,2)
                    

                    P = ratios.numel()
                    T1 = nom_s.shape[1]  # receding+1
                    if translations.size(0) != P:
                        raise ValueError(f"mosaic_translations first dim {translations.size(0)} != P {P}")


                    # translate then scale nom_s per poly
                    nom_s_b = nom_s.unsqueeze(0).expand(P, -1, -1).clone()       # (P,3,T1)
                    nom_s_b[:, :2, :] += translations.view(P, 2, 1)              # add per-poly translation
                    scaled_nom_s_b = nom_s_b.clone()
                    scaled_nom_s_b[:, :2, :] /= ratios.view(P, 1, 1)             # unit-space per poly

                    # scale obstacles + velocities per poly (stay in global frame, only scale xy)
                    if obs_points is not None:
                        scaled_obs_b = obs_points.unsqueeze(0).expand(P, -1, -1).clone()   # (P,2,N)
                        scaled_obs_b[:, :2, :] /= ratios.view(P, 1, 1)
                    else:
                        scaled_obs_b = None

                    if point_velocities is not None:
                        point_vel_b = point_velocities.unsqueeze(0).expand(P, -1, -1).clone()  # (P,2,N)
                        point_vel_b[:, :2, :] /= ratios.view(P, 1, 1)
                    else:
                        point_vel_b = None

                    # Keep a base call to preserve R_list (yaw-only) and dune_points behavior
                    _, R_base_list, obs_points_base_list = self.generate_point_flow(
                        nom_s, obs_points, point_velocities
                    )

                    # --- prepare all per-poly lists (lightweight loop, heavy NN will be single call) ---
                    per_poly_point_flow = []
                    per_poly_obs_points = []
                    for p in range(P):
                        pf_p, _, obs_p = self.generate_point_flow(
                            scaled_nom_s_b[p],
                            scaled_obs_b[p] if scaled_obs_b is not None else None,
                            point_vel_b[p]   if point_vel_b   is not None else None,
                        )
                        per_poly_point_flow.append(pf_p)  # list length T1
                        per_poly_obs_points.append(obs_p) # list length T1

                    # flatten across polys: length = P * T1
                    point_flow_big   = [pf for p in range(P) for pf in per_poly_point_flow[p]]
                    obs_points_big   = [op for p in range(P) for op in per_poly_obs_points[p]]

                    # R_list must be same length as point_flow_big; R is yaw-only (independent of scale/translation)
                    R_list_big = []
                    for _p in range(P):
                        if len(R_base_list) == T1:
                            R_list_big.extend(R_base_list)
                        elif len(R_base_list) == T1 - 1:
                            # pad last rotation for the final time step if your generate_point_flow returns T entries
                            R_list_big.extend(R_base_list + [R_base_list[-1]])
                        else:
                            raise ValueError(f"R_list length {len(R_base_list)} unexpected for T1={T1}")

                    # --- SINGLE dune call over all polys and times ---
                    mu_big, lam_big, sort_pts_big, dist_big = self.dune_layer(
                        point_flow_big, R_list_big, obs_points_big
                    )

                    # --- regroup back to poly-major and rescale geometric outputs ---
                    items_per_poly = T1
                    mu_list_mosaic, lam_list_mosaic = [], []
                    sort_point_list_mosaic, distance_list_mosaic = [], []

                    for p in range(P):
                        s = p * items_per_poly
                        e = (p + 1) * items_per_poly

                        mu_p  = mu_big[s:e]
                        lam_p = lam_big[s:e]
                        # rescale back from unit-space
                        r = ratios[p]
                        sort_p = [sp * r for sp in sort_pts_big[s:e]]
                        dist_p = [d  * r for d  in dist_big[s:e]]

                        mu_list_mosaic.append(mu_p)
                        lam_list_mosaic.append(lam_p)
                        sort_point_list_mosaic.append(sort_p)
                        distance_list_mosaic.append(dist_p)

                    # Keep poly-major structure
                    mu_list = mu_list_mosaic
                    lam_list = lam_list_mosaic
                    sort_point_list = sort_point_list_mosaic
                    distance_list = distance_list_mosaic

                    # min distance (original units)
                    mins = []
                    for poly in distance_list_mosaic:
                        for d in poly:
                            if isinstance(d, torch.Tensor) and d.numel() > 0:
                                mins.append(torch.min(d))
                    if mins:
                        self.min_distance = float(torch.stack(mins).min().item())

                    # dune_points property behavior unchanged: store base obstacle points at time 0 (global)
                    if obs_points_base_list and len(obs_points_base_list) > 0:
                        self._points = obs_points_base_list[0]


                else:
                    point_flow_list, R_list, obs_points_list = self.generate_point_flow(
                    nom_s, obs_points, point_velocities
                    )
                    mu_list, lam_list, sort_point_list, distance_list = self.forward_dune(
                        point_flow_list, R_list, obs_points_list
                    )
            else:
                mu_list, lam_list, sort_point_list, distance_list = [], [], [], []
                
            nom_s, nom_u, nom_distance = self.nrmp_layer(
                nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, sort_point_list, distance_list, actual_vel
            )
            if self.stop_criteria(nom_s, nom_u, mu_list, lam_list):
                break

        if configuration.log_cost:
            print(f"costs: {self.nrmp_layer.costs}")
            print("-" * 50)

        return nom_s, nom_u, nom_distance


    @time_it('- dune forward')
    def forward_dune(self, point_flow_list: list[torch.Tensor], R_list: list[torch.Tensor], obs_points_list: list[torch.Tensor]):
        if self.is_multipolygon:
            mu_list, lam_list, sort_point_list, distance_list = [], [], [], []
            for i, dune_layer in enumerate(self.dune_layer_list):
                mu_list_i, lam_list_i, sort_point_list_i, distance_list_i = dune_layer(
                    point_flow_list, R_list, obs_points_list
                )
                # if i==0:
                #     print(f"mu_list: {mu_list_i},\n lam_list: {lam_list_i}")
                #     print(f"sort_point_list: {sort_point_list_i},\n distance_list: {distance_list_i}")
                mu_list.append(mu_list_i)
                lam_list.append(lam_list_i)
                sort_point_list.append(sort_point_list_i)
                distance_list.append(distance_list_i)
        else:  
            mu_list, lam_list, sort_point_list, distance_list = self.dune_layer(
                point_flow_list, R_list, obs_points_list
            )

        return mu_list, lam_list, sort_point_list, distance_list


    def generate_point_flow(self, nom_s: torch.Tensor, obs_points: torch.Tensor, point_velocities: Optional[torch.Tensor]=None):

        '''
        generate the point flow (robot coordinate), rotation matrix and obs points (global coordinate) list in each receding step

        Args:
            nom_s: (3, receding+1)
            obs_points: (2, n)
            point_velocities: (2, n), x,y vel

        Returns:
            point_flow_list: list of (2, n); robot coordinate
            R_list: list of (2, 2); rotation matrix
            obs_points_list: list of (2, n); global coordinate
        '''

        # down sample the obs points by dune max num 

        if point_velocities is None:
            point_velocities = torch.zeros_like(obs_points)

        if obs_points.shape[1] > self.dune_max_num:
            self.print_once(f"down sample the obs points from {obs_points.shape[1]} to {self.dune_max_num}") 
            obs_points = downsample_decimation(obs_points, self.dune_max_num)
            point_velocities = downsample_decimation(point_velocities, self.dune_max_num)

        
        obs_points_list = []
        point_flow_list = []
        R_list = []

        if point_velocities is None:
            point_velocities = torch.zeros_like(obs_points)

        for i in range(self.T+1):
            receding_obs_points = obs_points + i * (point_velocities * self.dt)
            obs_points_list.append(receding_obs_points) 
            p0, R = self.point_state_transform(nom_s[:, i], receding_obs_points)
            point_flow_list.append(p0)
            R_list.append(R)

        return point_flow_list, R_list, obs_points_list


    def point_state_transform(self, state: torch.Tensor, obs_points: torch.Tensor):

        '''
        transform the position of obstacle points to the robot coordinate system in each receding step
        
        input: 
            state: [x, y, theta] -- transition and rotation matrix
            obs_points: (2, n) -- point cloud

        output:
            p0: (2, n) point cloud in the robot coordinate system
            R: (2, 2) rotation matrix
        '''

        state = state.reshape((3, 1))
        trans = state[0:2]
        theta = state[2, 0]
        R = to_device(torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]))

        p0 = R.T @ (obs_points - trans)

        return p0, R
    

    def stop_criteria(self, nom_s, nom_u, mu_list, lam_list):

        '''
        stop criteria for the iteration
        '''

        if self.current_nom_values[0] is None:
            self.current_nom_values = [nom_s, nom_u, mu_list, lam_list]
            return False
        
        else:

            nom_s_diff = torch.norm(nom_s - self.current_nom_values[0])
            nom_u_diff = torch.norm(nom_u - self.current_nom_values[1])

            if len(mu_list) == 0 or len(self.current_nom_values[2]) == 0:
                diff = nom_s_diff**2 + nom_u_diff**2

            else:
                if self.is_multipolygon or self.is_mosaic:
                    diff = 0
                    
                    for i in range(self.num_of_polygons):
                        effect_num = min([mu_list[i][0].shape[1], self.current_nom_values[2][i][0].shape[1], self.nrmp_max_num])
                        
                        mu_diff = torch.norm(torch.cat(mu_list[i])[:, :effect_num] - torch.cat(self.current_nom_values[2][i])[:, :effect_num]) / effect_num
                        lam_diff = torch.norm(torch.cat(lam_list[i])[:, :effect_num] - torch.cat(self.current_nom_values[3][i])[:, :effect_num]) / effect_num
                        
                        diff += mu_diff**2 + lam_diff**2

                else:
                    effect_num = min([mu_list[0].shape[1], self.current_nom_values[2][0].shape[1], self.nrmp_max_num])

                    mu_diff = torch.norm( (torch.cat(mu_list)[:, :effect_num] - torch.cat(self.current_nom_values[2])[:, :effect_num] )) / effect_num
                    lam_diff = torch.norm( (torch.cat(lam_list)[:, :effect_num]  - torch.cat(self.current_nom_values[3])[:, :effect_num]  )) / effect_num

                    diff = mu_diff**2 + lam_diff**2

            self.current_nom_values = [nom_s, nom_u, mu_list, lam_list]

            return diff < self.iter_threshold


    @property
    def min_distance(self):
        
        if self.no_obs:
            return inf
        elif self.is_multipolygon:
            return min([dune_layer.min_distance for dune_layer in self.dune_layer_list])
        elif self.is_mosaic:
            return self._min_distance
        else:
            return self.dune_layer.min_distance

    @min_distance.setter
    def min_distance(self, value):
        if self.is_mosaic:
            self._min_distance = value
        else:
            # This is to avoid setting min_distance for other types,
            # as it's a computed property for them.
            # You might want to raise an error or log a warning.
            pass

    @property
    def dune_points(self):
        
        if self.no_obs:
            return None
        elif self.is_multipolygon:
            return tensor_to_np(self.dune_layer_list[0].points)
        elif self.is_mosaic:
            return tensor_to_np(self._points) if self._points is not None else None
        else:
            return tensor_to_np(self.dune_layer.points)


    @property
    def nrmp_points(self):
        if self.nrmp_layer is None or self.no_obs:
            return None
        else:
            return tensor_to_np(self.nrmp_layer.points)
    

    def print_once(self, message):
        if not self.printed:
            print(message)
            self.printed = True
