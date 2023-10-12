import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion


class CoSLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()                # 建图的边界
        self.create_pose_data()             # 创建存储 估计的位姿 和 数据集中的位姿gt 到用的字典
        self.get_pose_representation()      # 查看当前数据集是用轴角还是四元数表示的,tum数据集是轴角
        self.keyframeDatabase = self.create_kf_database(config)                 # tum/fr1_desk: 每5帧取为一个关键帧
        
        # ! -------------------- 1. Scene representation -------------------- 
        self.model = JointEncoding(config, self.bounding_box).to(self.device)   # 得到encoding/decoding网络，用于获得深度和颜色信息
    
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)

    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)
    
    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose
 
    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))  # 从一个范围内的整数（0 到 H * W - 1）中随机选择samples个整数索引
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])
        
        return loss             

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        # ********************* 读取第0帧的相机位姿 *********************
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)       # [4,4]
        self.est_c2w_data[0] = c2w                  # 第0帧的观测位姿 直接作为 位姿估计
        self.est_c2w_data_rel[0] = c2w

        self.model.train()  # 将模型设置为训练模式

        # Training n_iters=100
        for i in range(n_iters):
            # ********************* 获得第0帧每个像素的颜色，深度，方向 *********************
            self.map_optimizer.zero_grad()      # 将之前的梯度信息清零
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])  # 从一个范围内的整数（0 到 H * W - 1）中随机选择samples=2048个样本像素

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)              # 得到每个采样得到的像素的h,w值             [2048]
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)       # 得到每个样本像素的方向，作为目标射线方向    [2048,3]
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)               # 得到每个样本像素的颜色，作为目标颜色        [2048.3]
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)  # 得到每个样本像素的深度，作为目标深度        [2048,1]

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)  # 世界坐标系下的射线原点，即变换矩阵的t，即相机位置 [2048,3]
            # rays_d_cam[..., None, :] 相机坐标系中的射线方向： [2048,1,3]
            # c2w[:3, :3] : 旋转矩阵    [3,3]
            # sum(x,-1) 在最后一个维度上进行求和
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)          # 世界坐标系下的射线方向 [2048,3]

            # Forward
            # ********************* 前向传播: 得到rgb图，深度图，rgb损失，深度损失,sdf损失，fs损失 *********************
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)    # 前向传播
            loss = self.get_loss_from_ret(ret)  # 将所有损失求和

            # ********************* 反响传播: 优化encoder/decoder网络的参数 *********************
            loss.backward()                     # 反响传播
            self.map_optimizer.step()           # 使用adam优化encoder和decoder网络的参数
        
        # ********************* 将当前帧加入关键帧 *********************
        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)
        
        print('First frame mapping done')
        return ret, loss    # 返回: rgb图，深度图，rgb损失，深度损失,sdf损失，fs损失 

    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')

        # ********************* 读取当前帧的位姿估计 *********************
        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()  # 将模型设置为训练模式

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            # ********************* 从数据集获得当前帧每个像素的颜色，深度，方向观测值 *********************
            self.cur_map_optimizer.zero_grad()  # 将之前的梯度信息清零
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

             # ********************* 根据位姿估计计算射线的原点和方向 *********************
            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # ********************* 前向传播: 得到rgb图，深度图，rgb损失，深度损失,sdf损失，fs损失 *********************
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)  # 将所有损失求和

            # ********************* 反响传播: 优化encoder/decoder网络的参数 *********************
            loss.backward()
            self.cur_map_optimizer.step()
        
        
        return ret, loss    # 返回: rgb图，深度图，rgb损失，深度损失,sdf损失，fs损失 

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    
    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])   # 提取变换矩阵的t
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3])) # 提取变换矩阵的R
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                               {"params": cur_trans, "lr": self.config[task]['lr_trans']}])
        
        return cur_rot, cur_trans, pose_optimizer
    
    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)


            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True)
            
            loss.backward(retain_graph=True)
            
            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
 
    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):                           # 第0帧已用于初始训练encoder和decoder网络
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)    # 此时，读取第0帧的位姿
            self.est_c2w_data[frame_id] = c2w_est_prev                      # 作为第1帧的位姿估计
            
        else:
            # ! -------------------- 2.2 Camera tracking: 初始化位姿估计 -------------------- 
            # 对于后面的帧使用论文的公式10,来初始化当前帧的位姿估计
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)   # 第i-2帧的位姿
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)        # 第i-1帧的位姿
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()            # T1 * T2^-1 
            self.est_c2w_data[frame_id] = delta@c2w_est_prev                    # T1 * T2^-1 * T1
        
        return self.est_c2w_data[frame_id]

    def tracking_pc(self, batch, frame_id):
        '''
        Tracking camera pose of current frame using point cloud loss
        (Not used in the paper, but might be useful for some cases)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)

        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config['tracking']['lr_rot']},
                                               {"params": cur_trans, "lr": self.config['tracking']['lr_trans']}])
        best_sdf_loss = None

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        thresh=0

        if self.config['tracking']['iter_point'] > 0:
            indice_pc = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['pc_samples'])
            rays_d_cam = batch['direction'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_s = batch['rgb'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_d = batch['depth'][:, iH:-iH, iW:-iW].reshape(-1, 1)[indice_pc].to(self.device)

            valid_depth_mask = ((target_d > 0.) * (target_d < 5.))[:,0]

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            for i in range(self.config['tracking']['iter_point']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)


                rays_o = c2w_est[...,:3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                pts_flat = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

                out = self.model.query_color_sdf(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:,:3])

                # TODO: Change this
                loss = 5 * torch.mean(torch.square(rgb-target_s)) + 1000 * torch.mean(torch.square(sdf))

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh +=1
                if thresh >self.config['tracking']['wait_iters']:
                    break

                loss.backward()
                pose_optimizer.step()
        

        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]


        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            # Not a keyframe, need relative pose
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        print('Best loss: {}, Camera loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
    
    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)        # 从数据集得到当前帧的位姿真值 [4, 4]

        # ********************* 初始化当前帧的位姿估计 *********************
        # Initialize current pose
        if self.config['tracking']['iter_point'] > 0:   # 通过点云损失来跟踪当前帧的相机位姿时使用，本论文没用该方法
            cur_c2w = self.est_c2w_data[frame_id]
        else:                                           # 使用论文的方法来跟踪相机位姿时使用
            # ! -------------------- 2.2 Camera tracking: 初始化位姿估计 -------------------- 
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])   

        # ********************* 一些训练变量 *********************
        indice = None
        best_sdf_loss = None
        thresh=0
        iW = self.config['tracking']['ignore_edge_W']   # 20
        iH = self.config['tracking']['ignore_edge_H']   # 20

        # ********************* 为位姿估计设置优化器 *********************
        # 提取当前位姿的t,R。并给他们创建Adam优化器
        # cur_rot: [1,3] 转为轴角表示
        # cur_trans: [1,3] 位移
        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)    


        # ! -------------------- 2.2 Camera tracking: 优化位姿 -------------------- 
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # ********************* 开始跟踪 *********************
        # Start tracking
        for i in range(self.config['tracking']['iter']):    # iter = 10

            pose_optimizer.zero_grad()  # 将R和t的adam优化器梯度设置为0

            # ********************* 将位姿估计：cur_rot, cur_trans转为变换矩阵 *********************
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)   # 轴角/四元素 转 变换矩阵 T。这里是轴角

            # ********************* 从当前帧中采样像素点,并从数据中读取这些像素点的观测值 *********************
            # Note here we fix the sampled points for optimisation
            if indice is None:
                # 从一个范围内的整数（0 到 H * W - 1）中随机选择samples=1024个图片像素点索引
                indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['sample'])
            
                # Slicing
                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)         # 将取样点的像素坐标(高和宽)提取出来
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device) # 相机坐标下各个采样点的射线方向 [1024, 3]
            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)             # 相机坐标下各个采样点的rgb观测值[1024, 3]
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)   # 相机坐标下各个采样点的深度观测值 [1024, 1]

            # ********************* 根据位姿估计计算射线起点和方向 *********************
            rays_o = c2w_est[...,:3, -1].repeat(self.config['tracking']['sample'], 1)   # 射线的原点：即估计位姿T中提取的位移t
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)       # 射线的方向：相机坐标下的射线方向 ✖ 位姿估计C2W 转为世界坐标系下的射线方向

            # ********************* 使用encoder/decoder网络，根据观测值和估计值计算损失函数 *********************
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)    # 得到rgb图，深度图，rgb损失，深度损失,sdf损失，fs损失
            loss = self.get_loss_from_ret(ret)                              # 所有的loss之和
            
            # ********************* 判断sdf损失有无变小,并找到最佳loss下的位姿估计 *********************
            if best_sdf_loss is None:               # 初始化best_sdf_loss
                best_sdf_loss = loss.cpu().item()   # 将loss的数据结构之和从tensor转为double
                best_c2w_est = c2w_est.detach()     # 创建张量c2w_est的无梯度副本
            with torch.no_grad(): # 下面代码块：禁用梯度计算
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)   # 将轴角cur_rot 和 位移cur_trans 转为 变换矩阵

                if loss.cpu().item() < best_sdf_loss:           # 如果损失相比之前变小了，就更新最好的sdf损失
                    best_sdf_loss = loss.cpu().item()           # 更新最好的sdf损失
                    best_c2w_est = c2w_est.detach()             # 创建张量c2w_est的无梯度副本,保存最佳loss下的位姿估计
                    thresh = 0
                else:
                    thresh +=1                                  # 如果优化后损失没比以前变小，thresh+1
            if thresh >self.config['tracking']['wait_iters']:   # wait_iters=100
                break
            
            # ********************* 更新参数 *********************
            loss.backward()             # 反响传播，计算相对于 cur_trans 和 cur_rot 的梯度
            pose_optimizer.step()       # 更新参数
        # ********************* 结束跟踪 *********************
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        # ********************* 使用loss最小的位姿估计作为当前帧的位姿估计 *********************
        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        # ! -------------------- 2.3 Tracked frame -------------------- 
        # Save relative pose of non-keyframes
        if frame_id % self.config['mapping']['keyframe_every'] != 0:        # 如果不是关键帧
            kf_id = frame_id // self.config['mapping']['keyframe_every']    # 前帧所属的关键帧的索引，比如11//5=2 第11帧属于第2个的关键帧
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']  # 关键帧id，比如2*5，第2个关键帧就是第10帧
            c2w_key = self.est_c2w_data[kf_frame_id]                        # 关键帧的估计位姿
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse() # 当前帧的T 乘上 关键帧的T^-1 得到两帧位姿之间的差异
            self.est_c2w_data_rel[frame_id] = delta                         # 保存差异
        
        print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
    
    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        
        # ********************* Optimizer for BA *********************
        # params: encoder/decoder网络的参数 
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
    
        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        # 创建 Adam 优化器，用于更新模型中的可训练参数（trainable_parameters）
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
        #********************* Optimizer for current frame mapping *********************
        if self.config['mapping']['cur_frame_iters'] > 0:
            params_cur_mapping = [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
                 
            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))
        
    
    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)      
        
    def run(self):
        # ********************* 创建map和BA的优化器 *********************
        # Adam 优化器，用于优化encoder/decoder网络
        # 优化位姿的优化器见tracking_render()
        self.create_optimizer()

        # ********************* 加载数据 *********************
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])


        # ! -------------------- 2/3. Start Co-SLAM(tracking + Mapping) -------------------- 
        """
        tqdm类:给迭代过程显示进度条。
        enumerate():同时获得可迭代对象的索引和对应元素,所以i是索引,batch是当前批次的数据。

        batch: tum数据集
            - frame_id: [i] 同索引i
            - c2w:      [1,4,4]      
            - rgb:      [1,368,496,3]
            - depth:    [1,368,496]
            - direction:[1,368,496,3]
        """ 
        for i, batch in tqdm(enumerate(data_loader)):
            # ********************* 可视化rgb和深度图 *********************
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)     # 将图片的颜色通道从BGR转为RGB        [368, 496, 3]
                raw_depth = batch["depth"]                                                      # 每一像素的深度作为射线深度           [1,368,496]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)              # 创建一个掩码，用于过滤小于5的深度     [368,496]
                depth_colormap = colormap_image(batch["depth"])                                 # 将1通道的深度图像转为3通道的颜色图    [3, 368, 496]
                depth_colormap[:, mask] = 255.                                                  # 用掩码将深度小于5的像素的rgb都设为255
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()                  # 换一下排列                         [368, 496, 3]
                image = np.hstack((rgb, depth_colormap))                                        # 水平方向上合并rgb图和深度图          [368, 992, 3]
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)                         # 在窗口中显示上面这2张图
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            # ********************* 建立初始的 地图和位姿估计 *********************
            # First frame mapping
            if i == 0:
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])  # ? 包含2.1
            
            # ********************* 建立每一帧的地图和位姿估计 *********************
            # Tracking + Mapping
            else:

                # ! -------------------- 2. tracking -------------------- 
                if self.config['tracking']['iter_point'] > 0:
                    # 通过点云损失来跟踪当前帧的相机位姿，本论文没用该方法
                    self.tracking_pc(batch, i)  
                    # 使用当前帧的rgb损失，深度损失，sdf损失，fs损失来跟踪当前帧的相机位姿
                self.tracking_render(batch, i)              # ? 包含2.1, 2.2, 2.3


                # ! -------------------- 3. Mapping -------------------- 
                if i%self.config['mapping']['map_every']==0:    # 每5帧建一次图
                    self.current_frame_mapping(batch, i)    # ? 包含3.2
                    # ! -------------------- 3.3 BA -------------------- 
                    self.global_BA(batch, i)                
                    # ! -------------------- 3.1 Keyframe database -------------------- 
                # Add keyframe
                if i % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('add keyframe:',i)
            

                # ! -------------------- * Evaluation -------------------- 
                if i % self.config['mesh']['vis']==0:
                    self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])
                    pose_relative = self.convert_relative_pose()
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
                    pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')
                    # 展示轨迹真值和预测轨迹图
                    if cfg['mesh']['visualisation']:
                        cv2.namedWindow('Traj:'.format(i), cv2.WINDOW_AUTOSIZE)
                        traj_image = cv2.imread(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "pose_r_{}.png".format(i)))
                        # best_traj_image = cv2.imread(os.path.join(best_logdir_scene, "pose_r_{}.png".format(i)))
                        # image_show = np.hstack((traj_image, best_traj_image))
                        image_show = traj_image
                        cv2.imshow('Traj:'.format(i), image_show)
                        key = cv2.waitKey(1)

        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i)) 
        
        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])
        
        pose_relative = self.convert_relative_pose()
        pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
        pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')

        #TODO: Evaluation of reconstruction


if __name__ == '__main__':
    
    # ********************* 加载参数 *********************
    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])    # save_path: "output/TUM/fr_desk/demo"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("coslam.py", os.path.join(save_path, 'coslam.py'))              

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))  # 将名为 cfg 的 Python 字典转换为每级缩进4个空格的 JSON 格式字符串。


    # ********************* 开始SLAM *********************
    # ! -------------------- 1. Scene representation: 网络构建  -------------------- 
    slam = CoSLAM(cfg)
    # ! -------------------- 2/3. Start Co-SLAM(tracking + Mapping) -------------------- 
    slam.run()
