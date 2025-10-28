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

        if not self.no_obs:
            if self.is_multipolygon and not self.is_mosaic:
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
                    robot.G,
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
                point_flow_list, R_list, obs_points_list = self.generate_point_flow(
                    nom_s, obs_points, point_velocities
                )
                mu_list, lam_list, sort_point_list, distance_list = self.forward_dune(
                    point_flow_list, R_list, obs_points_list
                )
                
                if self.is_mosaic:
                    T1 = self.T + 1
                    P = self.num_of_polygons
                    ratios = self.robot.mosaic_ratios

                    mu_list = [mu_list[i * T1 : (i + 1) * T1] for i in range(P)]
                    lam_list = [lam_list[i * T1 : (i + 1) * T1] for i in range(P)]
                    sort_point_list = [[pt * ratios[i] for pt in sort_point_list[i * T1 : (i + 1) * T1]] for i in range(P)]
                    distance_list = [[d * ratios[i] for d in distance_list[i * T1 : (i + 1) * T1]] for i in range(P)]

                    # Override DUNE's min_distance and obstacle_points with original scale
                    self.dune_layer.min_distance = torch.stack([dist_per_poly[0] for dist_per_poly in distance_list]).min()
                    self.dune_layer.obstacle_points = obs_points
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
            if self.is_mosaic:
                mu_list, lam_list, sort_point_list, distance_list = self.dune_layer.batch_forward(
                    point_flow_list, R_list, obs_points_list
                )
            else:
                mu_list, lam_list, sort_point_list, distance_list = [], [], [], []
                for i, dune_layer in enumerate(self.dune_layer_list):
                    mu_list_i, lam_list_i, sort_point_list_i, distance_list_i = dune_layer(
                        point_flow_list, R_list, obs_points_list
                    )
                    mu_list.append(mu_list_i)
                    lam_list.append(lam_list_i)
                    sort_point_list.append(sort_point_list_i)
                    distance_list.append(distance_list_i)
        else:
            mu_list, lam_list, sort_point_list, distance_list = self.dune_layer(
                point_flow_list, R_list, obs_points_list
            )

        return mu_list, lam_list, sort_point_list, distance_list

    # @time_it('- generate point flow')
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

        if self.is_mosaic:
            point_flow_list, R_list, obs_points_list = self.generate_point_flow_mosaic(nom_s, obs_points, point_velocities)
            return point_flow_list, R_list, obs_points_list
        
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
    

    def generate_point_flow_mosaic(self, nom_s: torch.Tensor, obs_points: torch.Tensor, point_velocities: Optional[torch.Tensor]=None):
        if not self.is_mosaic:
            raise ValueError("generate_point_flow_mosaic is only supported for mosaic robot")
        
        ratios = self.robot.mosaic_ratios
        translations = self.robot.mosaic_translations

        P = self.num_of_polygons
        T1 = self.T + 1
        assert nom_s.shape[1] == T1, f"nom_s must have shape (3, {T1})"
        
        # translate then scale nom_s per poly
        nom_s_b = nom_s.unsqueeze(0).expand(P, -1, -1).clone() # (P, 3, T1)
        nom_s_b[:, :2, :] += translations.view(P, 2, 1)
        scaled_nom_s_b = nom_s_b.clone() # (P, 3, T1)
        scaled_nom_s_b[:, :2, :] /= ratios.view(P, 1, 1)

        # scale obstacles + velocities per poly (stay in global frame, only scale xy)   
        scaled_obs_b = obs_points.unsqueeze(0).expand(P, -1, -1).clone() # (P, 2, N)
        scaled_obs_b[:, :2, :] /= ratios.view(P, 1, 1)

        if point_velocities is not None:
            point_vel_b = point_velocities.unsqueeze(0).repeat(P, 1, 1) # (P, 2, N)
            point_vel_b[:, :2, :] /= ratios.view(P, 1, 1)
        else:
            point_vel_b = None
        
        # Batch generate point flows for all polys at once
        batched_pf_list, _, batched_obs_list = self.generate_point_flow_batched(
            scaled_nom_s_b, scaled_obs_b, point_vel_b
        )

        # batched_pf_list: list of length T1, each (P,2,N)
        pf_stack = torch.stack(batched_pf_list, dim=0) # (T1, P, 2, N)
        pf_flat = pf_stack.permute(1, 0, 2, 3).reshape(P*T1, 2, -1) # (P*T1, 2, N)
        point_flow_list= list(pf_flat.unbind(0))
        
        # obs_points_list: list of length T1, each (P,2,N)
        obs_stack = torch.stack(batched_obs_list, dim=0) # (T1, P, 2, N)
        obs_flat = obs_stack.permute(1, 0, 2, 3).reshape(P*T1, 2, -1) # (P, T1, 2, N)
        obs_points_list = list(obs_flat.unbind(0))

        # Compute R_list directly from nom_s to avoid duplicates
        R_base_list = self.compute_R_list(nom_s)
        R_list = R_base_list * P
        
        return point_flow_list, R_list, obs_points_list


    def generate_point_flow_batched(self, nom_s: torch.Tensor, obs_points: torch.Tensor, point_velocities: Optional[torch.Tensor]=None):

        '''
        generate the point flow (robot coordinate), rotation matrix and obs points (global coordinate) list in each receding step

        Args:
            nom_s: (B, 3, receding+1)
            obs_points: (B, 2, n)
            point_velocities: (B, 2, n), x,y vel

        Returns:
            point_flow_list: list of (B, 2, n)
            R_list: list of (B, 2, 2)
            obs_points_list: list of (B, 2, n)
        '''

        if nom_s.dim() == 2:
            nom_s_b = nom_s.unsqueeze(0)
        elif nom_s.dim() == 3:
            nom_s_b = nom_s
        else:
            raise ValueError("nom_s must have ndim 2 or 3")

        if obs_points.dim() == 2:
            obs_b = obs_points.unsqueeze(0)
        elif obs_points.dim() == 3:
            obs_b = obs_points
        else:
            raise ValueError("obs_points must have ndim 2 or 3")

        if point_velocities is None:
            pv_b = torch.zeros_like(obs_b)
        else:
            if point_velocities.dim() == 2:
                pv_b = point_velocities.unsqueeze(0)
            elif point_velocities.dim() == 3:
                pv_b = point_velocities
            else:
                raise ValueError("point_velocities must have ndim 2 or 3")

        T1 = self.T + 1

        # Batch compute receding obstacle positions over time
        times = to_device(torch.arange(0, T1)).view(T1, 1, 1, 1)
        pv_dt_b = pv_b.unsqueeze(0) * self.dt # (1, B, 2, N)
        receding_obs = obs_b.unsqueeze(0) + times * pv_dt_b # (T1, B, 2, N)

        nom_perm = nom_s_b.permute(2, 0, 1).contiguous() # (T1, B, 3)
        trans = nom_perm[:, :, 0:2].unsqueeze(-1) # (T1, B, 2, 1)
        theta = nom_perm[:, :, 2] # (T1, B)

        c, s = torch.cos(theta), torch.sin(theta)
        R_batched = torch.stack(
            [torch.stack([c, -s], dim=-1),
             torch.stack([s, c], dim=-1)]
            , dim=-2
        ) # (T1, B, 2, 2)

        p0_batched = R_batched.transpose(-1, -2) @ (receding_obs - trans) # (T1, B, 2, N)

        point_flow_list = list(p0_batched.unbind(0))
        R_list = list(R_batched.unbind(0))
        obs_points_list = list(receding_obs.unbind(0))

        return point_flow_list, R_list, obs_points_list
                  
 
    def compute_R_list(self, nom_s: torch.Tensor):
        """
        Compute rotation matrices R_list from nom_s

        Args:
            nom_s: (B, 3, T1) or (3, T1)

        Returns:
            R_list: list length T1 of (B, 2, 2) or (2, 2)
        """
        if nom_s.dim() == 2:
            nom_s_b = nom_s.unsqueeze(0)
        elif nom_s.dim() == 3:
            nom_s_b = nom_s
        else:
            raise ValueError("nom_s must have ndim 2 or 3")

        nom_perm = nom_s_b.permute(2, 0, 1).contiguous() # (T1, B, 3)
        theta = nom_perm[:, :, 2] # (T1, B)

        c, s = torch.cos(theta), torch.sin(theta)
        R_batched = torch.stack(
            [torch.stack([c, -s], dim=-1),
             torch.stack([s, c], dim=-1)],
            dim=-2) # (T1, B, 2, 2)

        R_list = list(R_batched.unbind(0))
        if nom_s.dim() == 2:
            R_list = [R.squeeze(0) for R in R_list]

        return R_list


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
                if self.is_multipolygon:
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
        elif self.is_multipolygon and not self.is_mosaic:
            return min([dune_layer.min_distance for dune_layer in self.dune_layer_list])
        else:
            return self.dune_layer.min_distance


    @property
    def dune_points(self):
        
        if self.no_obs:
            return None
        elif self.is_multipolygon and not self.is_mosaic:
            return tensor_to_np(self.dune_layer_list[0].points)
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
