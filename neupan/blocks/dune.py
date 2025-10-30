"""
DUNE (Deep Unfolded Neural Encoder) is the core class of the PAN class. It maps the point flow to the latent distance space: mu and lambda. 

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

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
from math import inf
from neupan.blocks import ObsPointNet, DUNETrain
from neupan.configuration import np_to_tensor, to_device
from neupan.util import time_it, file_check, repeat_mk_dirs
from typing import Optional
import sys
class DUNE(torch.nn.Module):

    def __init__(self, receding: int=10, checkpoint =None, robot_G=None, robot_h=None, dune_max_num: int=100, train_kwargs: dict=dict(), robot_name=None, part_name=None) -> None:
        super(DUNE, self).__init__()
  
        self.T = receding
        self.max_num = dune_max_num

        self.robot_name = robot_name
        self.part_name = part_name

        self.G = np_to_tensor(robot_G)
        self.h = np_to_tensor(robot_h)
        self.edge_dim = self.G.shape[0]
        self.state_dim = self.G.shape[1]

        self.model = to_device(ObsPointNet(2, self.edge_dim))
        self.load_model(checkpoint, train_kwargs)

        self.obstacle_points = None
        self.min_distance = inf


        
    # @time_it('- dune forward')
    def forward(self, point_flow: list[torch.Tensor], R_list: list[torch.Tensor], obs_points_list: list[torch.Tensor]=[]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:

        '''
        map point flow to the latent distance features: lam, mu

        Args:
            point_flow: point flow under the robot coordinate, list of (state_dim, num_points); list length: T+1
            R_list: list of Rotation matrix, list of (2, 2), used to generate the lam from mu; list length: T+1
            obstacle_points_list: list of obstacle points, list of (2, num_points), global coordinate; list length: T+1

        Returns: 
            lam_list: list of lam tensor, each element is a tensor of shape (state_dim, num_points); list length: T+1
            mu_list: list of mu tensor, each element is a tensor of shape (edge_number, num_points); list length: T+1
            sort_point_list: list of point tensor, each element is a tensor of shape (state_dim, num_points); list length: T+1
            distance_list: list of distance tensor, each element is a tensor of shape (num_points,); list length: T+1;
        '''

        mu_list, lam_list, sort_point_list, distance_list = [], [], [], []
        self.obstacle_points = obs_points_list[0] # current obstacle points considered in the dune at time 0

        total_points = torch.hstack(point_flow)
        
        # map the point flow to the latent distance features mu
        with torch.no_grad():
            total_mu = self.model(total_points.T).T
        
        for index in range(self.T+1):
            num_points = point_flow[index].shape[1]
            mu = total_mu[:, index*num_points : (index+1)*num_points]
            R = R_list[index]
            p0 = point_flow[index]
            lam = (- R @ self.G.T @ mu)

            if mu.ndim == 1:
                mu = mu.unsqueeze(1)
                lam = lam.unsqueeze(1)

            distance = self.cal_objective_distance(mu, p0)

            if index == 0: 
                self.min_distance = torch.min(distance) 
            
            sort_indices = torch.argsort(distance)

            mu_list.append(mu[:, sort_indices])
            lam_list.append(lam[:, sort_indices])
            sort_point_list.append(obs_points_list[index][:, sort_indices])
            distance_list.append(distance[sort_indices])

        return mu_list, lam_list, sort_point_list, distance_list

    # @time_it('- dune forward')
    def batch_forward(self, point_flow: list[torch.Tensor], R_list: list[torch.Tensor], obs_points_list: list[torch.Tensor]=[]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:

        '''
        batched operation of mapping point flow to the latent distance features: lam, mu

        Args:
            point_flow: point flow under the robot coordinate, list of (state_dim, num_points); list length: N
            R_list: list of Rotation matrix, list of (2, 2), used to generate the lam from mu; list length: N
            obstacle_points_list: list of obstacle points, list of (2, num_points), global coordinate; list length: N

        Returns: 
            lam_list: list of lam tensor, each element is a tensor of shape (state_dim, num_points); list length: N
            mu_list: list of mu tensor, each element is a tensor of shape (edge_number, num_points); list length: N
            sort_point_list: list of point tensor, each element is a tensor of shape (state_dim, num_points); list length: N
            distance_list: list of distance tensor, each element is a tensor of shape (num_points,); list length: N;
        '''
       
        N = len(point_flow)
        assert N == len(R_list) == len(obs_points_list)
        
        self.obstacle_points = obs_points_list[0] # current obstacle points considered in the dune at time 0
        
        total_points = torch.cat([pf.T for pf in point_flow], dim=0) # (num_points, 2)
        # map the point flow to the latent distance features mu
        with torch.no_grad():
            total_mu = self.model(total_points).T
        
        point_sizes = [pf.shape[1] for pf in point_flow]
        P = max(point_sizes)      
                
        cols = to_device(torch.arange(P).unsqueeze(0)).expand(N, P)
        mask = cols < to_device(torch.as_tensor(point_sizes).unsqueeze(1)) # (N, P)
                
        point_flow_b = point_flow[0].new_zeros((N, 2, P)) # (N, 2, P)
        obs_points_b = obs_points_list[0].new_zeros((N, 2, P)) # (N, 2, P)
        mu_b = total_mu.new_zeros((self.edge_dim, N, P)) # (E, N, P)
        
        row_idx = to_device(torch.arange(N).unsqueeze(1).expand(N, P))[mask]
        col_idx = cols[mask]
        
        point_flow_b[row_idx, :, col_idx] = total_points # (num_points, 2)
        obs_points_b[row_idx, :, col_idx] = torch.cat([op.T for op in obs_points_list], dim=0) # (num_points, 2)
        mu_b[:, row_idx, col_idx] = total_mu
        
        # lam = - R @ G^T @ mu
        temp = torch.einsum('i j, j t p -> i t p', -self.G.T, mu_b) # (2, N, P)
        R = torch.stack(R_list, dim=0) # (N, 2, 2)
        lam_b = torch.bmm(R, temp.permute(1, 0, 2)) # (N, 2, P)

        distance_b = self.cal_objective_distance_batch(mu_b, point_flow_b) # (N, P)
                
        # a trick to sort only valid points
        very_neg = distance_b.new_full((), -1e30)
        row_max = torch.where(mask, distance_b, very_neg).amax(dim=1, keepdim=True)
        
        row_mean_abs = (distance_b.abs() * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)        
        big = row_max + row_mean_abs + 1
        distance_masked = torch.where(mask, distance_b, big)
        
        sorted_idx_b = torch.argsort(distance_masked, dim=1) # (N, P)
        sorted_distance_b = torch.gather(distance_b, 1, sorted_idx_b) # (N, P)

        # (E, N, P) -> (N, E, P)
        sorted_mu_b = torch.gather(mu_b.transpose(0, 1), 2, sorted_idx_b.unsqueeze(1).expand(-1, self.edge_dim, -1))
        # (N, 2, P) -> (N, 2, P)
        sorted_lam_b = torch.gather(lam_b, 2, sorted_idx_b.unsqueeze(1).expand(-1, 2, -1))
        # (N, 2, P) -> (N, 2, P)
        sorted_obs_points_b = torch.gather(obs_points_b, 2, sorted_idx_b.unsqueeze(1).expand(-1, 2, -1))
        
        mu_list, lam_list, sort_point_list, distance_list = [], [], [], []        
        for i, s in enumerate(point_sizes):               
            mu_list.append(sorted_mu_b[i, :, :s])
            lam_list.append(sorted_lam_b[i, :, :s])
            sort_point_list.append(sorted_obs_points_b[i, :, :s])
            distance_list.append(sorted_distance_b[i, :s])
            
            if i == 0:
                self.min_distance = sorted_distance_b[i, :s].min()
        
        return mu_list, lam_list, sort_point_list, distance_list


    def cal_objective_distance(self, mu: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:

        '''
        input: 
            mu: (edge_dim, num_points)
            p0: (state_dim, num_points)   
        output:
            distance:  mu.T (G @ p0 - h),  (num_points,)
        ''' 

        temp = (self.G @ p0 - self.h).T.unsqueeze(2)
        muT = mu.T.unsqueeze(1)

        distance = torch.squeeze(torch.bmm(muT, temp)) 

        if distance.ndim == 0:
            distance = distance.unsqueeze(0)

        return distance
    
    
    def cal_objective_distance_batch(self, mu: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:

        '''
        input: 
            mu: (edge_dim, N, num_points)
            p0: (N, state_dim, num_points)   
        output:
            distance:  mu.T (G @ p0 - h),  (N, num_points)
        ''' 

        temp = torch.einsum('e i, t i p -> e t p', self.G, p0)
        temp = temp - self.h.view(-1, 1, 1) # (edge_dim, N, num_points)

        distance = (mu * temp).sum(dim=0) # (N, num_points)

        return distance


    def load_model(self, checkpoint: Optional[str]=None, train_kwargs: Optional[dict]=None):

        '''
        checkpoint: pth file path of the model
        '''

        try:
            if checkpoint is None:
                raise FileNotFoundError

            self.abs_checkpoint_path = file_check(checkpoint)
            self.model.load_state_dict(torch.load(self.abs_checkpoint_path, map_location=torch.device('cpu')))
            to_device(self.model)
            self.model.eval()

        except FileNotFoundError:

            if train_kwargs is None or len(train_kwargs) == 0:
                print('No train kwargs provided. Default value will be used.')
                train_kwargs = dict()
            
            direct_train = train_kwargs.get('direct_train', False)

            if direct_train:
                print('train or test the model directly.')
                return 

            if self.ask_to_train():
                self.train_dune(train_kwargs)

                if self.ask_to_continue():
                    self.model.load_state_dict(torch.load(self.full_model_name, map_location=torch.device('cpu')))
                    to_device(self.model)
                    self.model.eval()
                else:
                    print('You can set the new model path to the DUNE class to use the trained model.') 

            else:
                print('Can not find checkpoint. Please check the path or train first.')
                raise FileNotFoundError


    def train_dune(self, train_kwargs):

        model_name = train_kwargs.get("model_name", self.robot_name)

        if self.part_name is not None:
            checkpoint_path = sys.path[0] + '/model' + '/' + model_name + '/' + self.part_name
        else:
            checkpoint_path = sys.path[0] + '/model' + '/' + model_name
        checkpoint_path = repeat_mk_dirs(checkpoint_path)
        
        self.train_model = DUNETrain(self.model, self.G, self.h, checkpoint_path)
        self.full_model_name = self.train_model.start(**train_kwargs)
        print('Complete Training. The model is saved in ' + self.full_model_name)

    def ask_to_train(self):
        
        while True:
            choice = input("Do not find the DUNE model; Do you want to train the model now, input Y or N:").upper()
            if choice == 'Y':
                return True
            elif choice == 'N':
                print('Please set the your model path for the DUNE layer.')
                sys.exit()
            else:
                print("Wrong input, Please input Y or N.")


    def ask_to_continue(self):
        
        while True:
            choice = input("Do you want to continue the case running, input Y or N:").upper()
            if choice == 'Y':
                return True
            elif choice == 'N':
                print('exit the case running.')
                sys.exit()
            else:
                print("Wrong input, Please input Y or N.")


    @property
    def points(self):
        '''
        point considered in the dune layer
        '''

        return self.obstacle_points

