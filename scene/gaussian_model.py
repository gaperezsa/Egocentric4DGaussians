#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# from utils.point_utils import addpoint, combine_pointcloud, downsample_point_cloud_open3d, find_indices_in_A
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        def steep_sigmoid(x):
            return torch.sigmoid(x)
        def inv_steep_sigmoid(x):
            return inverse_sigmoid(x)
        
        self.opacity_activation = steep_sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._dynamic_xyz = torch.empty(0).type(torch.bool)
        # self._deformation =  torch.empty(0)
        self._deformation = deform_network(args)
        # self.grid = TriPlaneGrid()
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.last_seen_means_scale_rot_opacity_shs = {}
        self.setup_functions()
        

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._dynamic_xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz,
        self._dynamic_xyz,
        deform_state,
        self._deformation_table,
        
        # self.grid,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_dynamic_xyz(self):
        return self._xyz[self._dynamic_xyz]

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._dynamic_xyz = torch.zeros(self._xyz.shape[0]).type(torch.bool).to(self._xyz.device)
        self._deformation = self._deformation.to("cuda") 
        # self.grid = self.grid.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)

    def erase_non_dynamic_grads(self):
        self._xyz.grad[~self._dynamic_xyz] *= 0
        self._features_dc.grad[~self._dynamic_xyz] *= 0
        self._features_rest.grad[~self._dynamic_xyz] *= 0
        self._opacity.grad[~self._dynamic_xyz] *= 0
        self._scaling.grad[~self._dynamic_xyz] *= 0
        self._rotation.grad[~self._dynamic_xyz] *= 0

    def training_setup(self, training_args, stage):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        
        if stage in ("background_depth","background_RGB"):
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "background_xyz"},
                {'params': list(self._deformation.get_mlp_parameters()), 'lr': 0, "name": "deformation"},
                {'params': list(self._deformation.get_grid_parameters()), 'lr': 0, "name": "grid"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
            max_steps = max(training_args.position_lr_max_steps, training_args.background_RGB_iterations, training_args.background_depth_iterations)
            self.background_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=max_steps)
            self.deformation_scheduler_args = get_expon_lr_func(lr_init=0,
                                                    lr_final=0,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)
            self.grid_scheduler_args = get_expon_lr_func(lr_init=0,
                                                    lr_final=0,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)
        
        
        elif stage == "dynamics_depth":
            # 1) sanity check
            N = self._xyz.shape[0]
            assert self._dynamic_xyz.shape[0] == N, \
                f"dynamic mask ({self._dynamic_xyz.shape[0]}) ≠ xyz ({N})"

            # 2) Install a hook function which adapts to the size of the dynamic mask
            def _dynamic_grad_hook(grad):
                # grad: [N, 3]
                m = self._dynamic_xyz.view(-1, 1).to(grad.device).to(grad.dtype)  # [N,1]
                return grad * m   # zero out any grads where m==0

            # Only register once per Parameter object
            if not getattr(self._xyz, "_has_dynamic_hook", False):
                self._xyz.register_hook(_dynamic_grad_hook)
                self._xyz._has_dynamic_hook = True

            # now safe to index
            l = [
                {'params': [self._xyz],'lr': training_args.position_lr_init * self.spatial_lr_scale,"name": "dynamic_xyz"},
                {'params': list(self._deformation.get_mlp_parameters()),'lr': training_args.deformation_lr_init * self.spatial_lr_scale,"name": "deformation"},
                {'params': list(self._deformation.get_grid_parameters()),'lr': training_args.grid_lr_init * self.spatial_lr_scale,"name": "grid"},
                {'params': [self._features_dc],   'lr': 0, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': 0, "name": "f_rest"},
                {'params': [self._opacity],       'lr': 0, "name": "opacity"},
                {'params': [self._scaling],       'lr': 0, "name": "scaling"},
                {'params': [self._rotation],      'lr': 0, "name": "rotation"},
            ]
            max_steps = max(training_args.position_lr_max_steps, training_args.dynamics_depth_iterations)
            self.dynamic_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=max_steps)
            self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)  
            self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)

        elif stage == "dynamics_RGB":

            # 1) sanity check
            N = self._xyz.shape[0]
            assert self._dynamic_xyz.shape[0] == N, \
                f"dynamic mask ({self._dynamic_xyz.shape[0]}) ≠ xyz ({N})"

            # 2) Install a hook function which adapts to the size of the dynamic mask
            def _dynamic_grad_hook(grad):
                # grad: [N, 3]
                m = self._dynamic_xyz.view(-1, 1).to(grad.device).to(grad.dtype)  # [N,1]
                return grad * m   # zero out any grads where m==0

            # Only register once per Parameter object
            if not getattr(self._xyz, "_has_dynamic_hook", False):
                self._xyz.register_hook(_dynamic_grad_hook)
                self._xyz._has_dynamic_hook = True

            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "dynamic_xyz"},
                {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
                {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
            max_steps = max(training_args.position_lr_max_steps, training_args.dynamics_RGB_iterations)
            self.dynamic_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=max_steps)
            self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)  
            self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)


        elif stage == "fine_coloring":
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler, "name": "dynamic_xyz"},
                {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler, "name": "deformation"},
                {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler, "name": "grid"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
            max_steps = max(training_args.position_lr_max_steps, training_args.fine_iterations)
            self.background_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=max_steps)
            self.dynamic_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=max_steps)
            self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler,
                                                    lr_final=training_args.deformation_lr_final * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)  
            self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler,
                                                    lr_final=training_args.grid_lr_final * self.spatial_lr_scale * training_args.fine_opt_dyn_lr_downscaler,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=max_steps)



        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "background_xyz":
                lr = self.background_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if param_group["name"] == "dynamic_xyz":
                lr = self.dynamic_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    def compute_deformation(self,time):
        
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz
    # def save_ply_dynamic(path):
    #     for time in range(self._deformation.shape(-1)):
    #         xyz = self.compute_deformation(time)
    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(self._deformation.deformation_net.grid.)
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._dynamic_xyz = torch.zeros(self._xyz.shape[0]).type(torch.bool).to(self._xyz.device)
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, keep_mask: torch.Tensor, dyn_mask: torch.Tensor):
        optimizable_tensors = {}
        total     = keep_mask.shape[0]
        dyn_count = int(dyn_mask.sum().item())
        bg_count  = total - dyn_count
        surv_count = int(keep_mask.sum().item())

        for group in self.optimizer.param_groups:
            # skip nets with >1 param (MLP, grid)
            if len(group["params"]) > 1:
                continue

            name      = group["name"]
            old_param = group["params"][0]
            M         = old_param.shape[0]

            # pick the exact slice
            if   M == total:     submask = keep_mask
            elif M == dyn_count: submask = keep_mask[dyn_mask]
            elif M == bg_count:  submask = keep_mask[~dyn_mask]
            elif M == surv_count:
                # already matches post-prune survivors
                submask = torch.ones((M,), dtype=torch.bool, device=old_param.device)
            else:
                raise RuntimeError(
                    f"Group '{name}' len={M}; total={total}, dyn={dyn_count}, "
                    f"bg={bg_count}, surv={surv_count}"
                )

            state = self.optimizer.state.get(old_param, None)
            if state is not None:
                state["exp_avg"]    = state["exp_avg"][submask]
                state["exp_avg_sq"] = state["exp_avg_sq"][submask]
                del self.optimizer.state[old_param]
                new_param = nn.Parameter(old_param[submask].requires_grad_(True))
                self.optimizer.state[new_param] = state
            else:
                new_param = nn.Parameter(old_param[submask].requires_grad_(True))

            group["params"][0]        = new_param
            optimizable_tensors[name] = new_param

        return optimizable_tensors


    def prune_points(self, mask: torch.Tensor):
        valid   = ~mask
        old_dyn = self._dynamic_xyz.clone()
        old_xyz = self._xyz.clone()

        # 0) Remember if we already had the dynamic hook
        had_hook = getattr(self._xyz, "_has_dynamic_hook", False)

        # 1) Prune the optimizer
        optim_tensors = self._prune_optimizer(valid, old_dyn)

        # 2) Shrink the dynamic flag
        self._dynamic_xyz = old_dyn[valid]

        # 3) Rebuild xyz
        surv_dyn = old_dyn & valid
        surv_bg  = (~old_dyn) & valid
        pruned_dyn = old_xyz[surv_dyn]
        pruned_bg  = old_xyz[surv_bg]

        N_surv = valid.sum().item()
        D      = old_xyz.size(1)
        new_xyz = torch.empty((N_surv, D), device=old_xyz.device)

        dyn_pos = old_dyn[valid]
        bg_pos  = ~old_dyn[valid]

        if "dynamic_xyz" in optim_tensors:
            new_xyz[dyn_pos] = optim_tensors["dynamic_xyz"]
        else:
            new_xyz[dyn_pos] = pruned_dyn

        if "background_xyz" in optim_tensors:
            new_xyz[bg_pos] = optim_tensors["background_xyz"]
        else:
            new_xyz[bg_pos] = pruned_bg

        # 4) Re‐wrap self._xyz (leaf + hook)
        self._xyz = nn.Parameter(new_xyz, requires_grad=True)

        # 5) Reassign all other per-point buffers
        self._features_dc   = optim_tensors["f_dc"]
        self._features_rest = optim_tensors["f_rest"]
        self._opacity       = optim_tensors["opacity"]
        self._scaling       = optim_tensors["scaling"]
        self._rotation      = optim_tensors["rotation"]
        self._deformation_table = self._deformation_table[valid]

        # 6) Reset accumulators
        n = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((n,1), device=self._xyz.device)
        self._deformation_accum = torch.zeros((n,3), device=self._xyz.device)
        self.denom              = torch.zeros((n,1), device=self._xyz.device)
        self.max_radii2D        = torch.zeros((n,),  device=self._xyz.device)

        # 7) If we had a hook, re‐install it now
        if had_hook:
            self._xyz._has_dynamic_hook = False
            def _dynamic_grad_hook(grad):
                m = self._dynamic_xyz.view(-1,1).to(grad.device).to(grad.dtype)
                return grad * m
            self._xyz.register_hook(_dynamic_grad_hook)
            self._xyz._has_dynamic_hook = True








    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self,
        new_xyz: torch.Tensor,
        new_dynamic_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_deformation_table: torch.Tensor
    ):
        # 0) Remember if we already had the dynamic hook on _xyz
        had_hook = getattr(self._xyz, "_has_dynamic_hook", False)

        # 1) Expand optimizer state exactly as before
        d = {
        "dynamic_xyz":    new_xyz[new_dynamic_xyz],
        "background_xyz": new_xyz[~new_dynamic_xyz],
        "f_dc":           new_features_dc,
        "f_rest":         new_features_rest,
        "opacity":        new_opacities,
        "scaling":        new_scaling,
        "rotation":       new_rotation,
        }
        self.cat_tensors_to_optimizer(d)

        # 2) Append & re-wrap every Parameter buffer

        # xyz
        cat_xyz = torch.cat([self._xyz.data, new_xyz], dim=0)
        self._xyz = nn.Parameter(cat_xyz, requires_grad=True)

        # features_dc
        cat_fdc = torch.cat([self._features_dc.data, new_features_dc], dim=0)
        self._features_dc = nn.Parameter(cat_fdc, requires_grad=True)

        # features_rest
        cat_frest = torch.cat([self._features_rest.data, new_features_rest], dim=0)
        self._features_rest = nn.Parameter(cat_frest, requires_grad=True)

        # opacity
        cat_op = torch.cat([self._opacity.data, new_opacities], dim=0)
        self._opacity = nn.Parameter(cat_op, requires_grad=True)

        # scaling
        cat_sc = torch.cat([self._scaling.data, new_scaling], dim=0)
        self._scaling = nn.Parameter(cat_sc, requires_grad=True)

        # rotation
        cat_rot = torch.cat([self._rotation.data, new_rotation], dim=0)
        self._rotation = nn.Parameter(cat_rot, requires_grad=True)

        # deformation_table & dynamic_xyz (plain tensors)
        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], dim=0)
        self._dynamic_xyz       = torch.cat([self._dynamic_xyz,       new_dynamic_xyz],       dim=0)

        # 3) Zero‐pad accumulators
        M = new_xyz.shape[0]
        dev = self._xyz.device
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum,
                                            torch.zeros((M,1), device=dev)], dim=0)
        self._deformation_accum = torch.cat([self._deformation_accum,
                                            torch.zeros((M,3), device=dev)], dim=0)
        self.denom              = torch.cat([self.denom,
                                            torch.zeros((M,1), device=dev)], dim=0)
        self.max_radii2D        = torch.cat([self.max_radii2D,
                                            torch.zeros((M,),  device=dev)], dim=0)

        # 4) If we had installed the dynamic hook, re‐install it now
        if had_hook:
            # clear previous flag on old tensor
            self._xyz._has_dynamic_hook = False
            # re‐register the hook
            def _dynamic_grad_hook(grad):
                m = self._dynamic_xyz.view(-1,1).to(grad.device).to(grad.dtype)
                return grad * m
            self._xyz.register_hook(_dynamic_grad_hook)
            self._xyz._has_dynamic_hook = True




    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = (padded_grad >= grad_threshold) & (~self._dynamic_xyz)

        # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_dynamic_xyz = self._dynamic_xyz[selected_pts_mask].repeat(N)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_dynamic_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        
        self.prune_points(prune_filter)

    def densify_and_split_dynamic(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
        median_dist: float = 0.1,
        factor: float = 0.1
    ):
        """
        Very‐local, Chamfer‐adaptive splitting of dynamic Gaussians.

        Args:
            grads (Tensor): 1D tensor of shape (M,) holding per-Gaussian gradient magnitudes.
            grad_threshold (float): threshold above which to consider splitting.
            scene_extent (float): used for any additional size checks (unused here).
            N (int): how many children to spawn per selected parent.
            median_dist (float): median chamfer distance between all dynamic Gaussians and GT points.
            factor (float): fraction of median_dist to use as sampling std.

        For each parent Gaussian i where:
            - grads[i] ≥ grad_threshold, and
            - parent is marked dynamic, and
            - parent’s scale > percent_dense * scene_extent
        we:
            1) set std = factor * median_dist,
            2) sample N offsets ∼ N(0, std) in world coords,
            3) position children = parent_center + offset,
            4) prune any child whose ‖offset‖ > median_dist,
            5) copy parent’s features, opacity, scale, rotation, deformation_flag to each surviving child,
            and mark them dynamic.
        """
        device = self._xyz.device
        n_pts = self.get_xyz.shape[0]

        # 1) Build a full‐length “padded” gradient vector
        padded = torch.zeros((n_pts,), device=device)
        # grads might be length M ≤ n_pts; copy into front
        padded[:grads.shape[0]] = grads.squeeze()

        # 2) Select parents: (grad ≥ thr) AND dynamic AND large enough to split
        sel = (padded >= grad_threshold) & self._dynamic_xyz
        sel = sel & (torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        if not sel.any():
            return

        # 3) Gather parent data (K = number of selected parents)
        parent_xyz     = self._xyz[sel]            # [K, 3]
        parent_scales  = self.get_scaling[sel]     # [K, 3]
        parent_radius  = torch.max(parent_scales, dim=1).values  # [K]

        K = parent_xyz.shape[0]
        # If median_dist is zero (unlikely), skip
        if median_dist <= 1e-9:
            return

        # 4) Compute global σ and maximum allowed
        offset_std = median_dist * factor            # a small fraction of median Chamfer distance
        max_allowed = 2 * median_dist            # children further than this get pruned

        # 5) Sample N children per parent: isotropic world-space
        #    Total to sample = K * N
        total_new = K * N
        # Create a (total_new × 3) tensor of isotropic offsets
        offsets = torch.normal(
            mean=torch.zeros((total_new, 3), device=device),
            std=torch.ones((total_new, 3), device=device) * offset_std
        )
        # Repeat each parent center N times → [K*N, 3]
        centers = parent_xyz.repeat(N, 1)

        candidate_xyz = centers + offsets  # [K*N, 3]

        # 6) Compute each child’s distance from its parent
        parent_repeated = parent_xyz.repeat(N, 1)  # [K*N, 3]
        dists = torch.norm(candidate_xyz - parent_repeated, dim=1)  # [K*N]

        # 7) Keep only those within max_allowed
        keep = (dists <= max_allowed)
        if not keep.any():
            return

        new_xyz     = candidate_xyz[keep]                     # [M, 3]
        new_dynamic = torch.ones(keep.sum(), dtype=torch.bool, device=device)

        # 8) Copy parent attributes, repeated N times, then indexed by keep
        #    First gather parent tensors (each length K)
        fe_dc_parent    = self._features_dc[sel]        # [K, F_dc, 1]
        fe_r_parent     = self._features_rest[sel]      # [K, F_rest, 1]
        op_parent       = self._opacity[sel]            # [K, 1]
        sc_parent       = self._scaling[sel]            # [K, 3]
        rot_parent      = self._rotation[sel]           # [K, 4]
        def_tbl_parent  = self._deformation_table[sel]  # [K]

        # Helper to repeat each parent entry N times, then mask by keep
        def repeat_and_keep(x: torch.Tensor):
            # x.shape = [K, ...]. We want [K*N, ...], repeated along batch dim
            rep = x.repeat_interleave(N, dim=0)  # [K*N, ...]
            return rep[keep]

        new_features_dc   = repeat_and_keep(fe_dc_parent)   # [M, F_dc, 1]
        new_features_rest = repeat_and_keep(fe_r_parent)    # [M, F_rest, 1]
        new_opacity       = repeat_and_keep(op_parent)      # [M, 1]
        new_scaling       = repeat_and_keep(sc_parent)      # [M, 3]
        new_rotation      = repeat_and_keep(rot_parent)     # [M, 4]
        new_deform_tbl    = repeat_and_keep(def_tbl_parent) # [M]

        # 9) Finally, add them into the Gaussian model
        self.densification_postfix(
            new_xyz,
            new_dynamic,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_deform_tbl
        )

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # 主动增加稀疏点云
        # if not hasattr(self,"voxel_size"):
        #     self.voxel_size = 8  
        # if not hasattr(self,"density_threshold"):
        #     self.density_threshold = density_threshold
        # if not hasattr(self,"displacement_scale"):
        #     self.displacement_scale = displacement_scale
        # point_cloud = self.get_xyz.detach().cpu()
        # sparse_point_mask = self.downsample_point(point_cloud)
        # _, low_density_points, new_points, low_density_index = addpoint(point_cloud[sparse_point_mask],density_threshold=self.density_threshold,displacement_scale=self.displacement_scale,iter_pass=0)
        # sparse_point_mask = sparse_point_mask.to(grads_accum_mask)
        # low_density_index = low_density_index.to(grads_accum_mask)
        # if new_points.shape[0] < 100 :
        #     self.density_threshold /= 2
        #     self.displacement_scale /= 2
        #     print("reduce diplacement_scale to: ",self.displacement_scale)
        # global_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool).to(grads_accum_mask)
        # global_mask[sparse_point_mask] = low_density_index
        # selected_pts_mask_grow = torch.logical_and(global_mask, grads_accum_mask)
        # print("降采样点云:",sparse_point_mask.sum(),"选中的稀疏点云：",global_mask.sum(),"梯度累计点云：",grads_accum_mask.sum(),"选中增长点云：",selected_pts_mask_grow.sum())
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # breakpoint()        
        new_xyz = self._xyz[selected_pts_mask] 
        new_dynamic_xyz = self._dynamic_xyz[selected_pts_mask.to(self._dynamic_xyz.device)]
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        # if opt.add_point:
        # selected_xyz, grow_xyz = self.add_point_by_mask(selected_pts_mask_grow.to(self.get_xyz.device), self.displacement_scale)
        self.densification_postfix(new_xyz, new_dynamic_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)
        # print("被动增加点云：",selected_xyz.shape[0])
        # print("主动增加点云：",selected_pts_mask.sum())
        # if model_path is not None and iteration is not None:
        #     point = combine_pointcloud(self.get_xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy(), selected_xyz.detach().cpu().numpy())
        #     write_path = os.path.join(model_path,"add_point_cloud")
        #     os.makedirs(write_path,exist_ok=True)
        #     o3d.io.write_point_cloud(os.path.join(write_path,f"iteration_{stage}{iteration}.ply"),point)
        #     print("write output.")

    def spawn_dynamic_gaussians(self, selected_pts_mask=None):
        # 1) Select a random subset if not provided
        if selected_pts_mask is None:
            selected_pts_mask = torch.rand(self._xyz.shape[0], device=self._xyz.device) < 0.01
        else:
            assert len(selected_pts_mask) == self._xyz.shape[0]

        n_new = selected_pts_mask.sum().item()
        if n_new == 0:
            return

        # 2) Compute “standard” size & opacity from the existing set
        #    - median scaling along each axis
        median_scale = torch.median(self._scaling, dim=0).values  # shape: (3,)
        #    - maximum opacity
        max_opacity = self._opacity.max().item()                  # scalar
        # 3) Gather the positions & rotations & features for the selected centers
        new_xyz            = self._xyz[selected_pts_mask]
        new_dynamic_xyz    = torch.ones(n_new, dtype=torch.bool, device=self._xyz.device)
        new_features_dc    = self._features_dc[selected_pts_mask]
        new_features_rest  = self._features_rest[selected_pts_mask]
        new_rotation       = self._rotation[selected_pts_mask]
        new_deformation_tbl= self._deformation_table[selected_pts_mask]

        # 4) Override their scale & opacity to be uniform
        #    - expand median_scale from (3,) → (n_new, 3)
        new_scaling = median_scale.unsqueeze(0).repeat(n_new, 1)
        #    - create an (n_new, 1) tensor at max_opacity
        new_opacities = torch.full((n_new, 1), max_opacity, device=self._xyz.device)

        # 5) Append them into your model’s buffers
        self.densification_postfix(
            new_xyz,
            new_dynamic_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_deformation_tbl
        )

    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    def get_displayment(self,selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]
    
        # while (mask_d.sum()/final_point.shape[0])<0.5:
        #     perturb/=2
        #     displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        #     final_point = selected_point + displacements
        #     mask_a = final_point<xyz_max 
        #     mask_b = final_point>xyz_min
        #     mask_c = mask_a & mask_b
        #     mask_d = mask_c.all(dim=1)
        #     final_point = final_point[mask_d]
        return final_point, mask_d    
    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)
        # displacements = torch.randn(selected_xyz.shape[0], 3).to(self._xyz) * perturb

        # new_xyz = selected_xyz + displacements
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_dynamic_xyz = self._dynamic_xyz[selected_pts_mask][mask]
        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_dynamic_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)
        return selected_xyz, new_xyz


    def downsample_point(self, point_cloud):
        if not hasattr(self,"voxel_size"):
            self.voxel_size = 8  
        point_downsample = point_cloud
        flag = False 
        while point_downsample.shape[0]>1000:
            if flag:
                self.voxel_size+=8
            point_downsample = downsample_point_cloud_open3d(point_cloud,voxel_size=self.voxel_size)
            flag = True
        print("point size:",point_downsample.shape[0])
        # downsampled_point_mask = torch.eq(point_downsample.view(1,-1,3), point_cloud.view(-1,1,3)).all(dim=1)
        downsampled_point_index = find_indices_in_A(point_cloud, point_downsample)
        downsampled_point_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool).to(point_downsample.device)
        downsampled_point_mask[downsampled_point_index]=True
        return downsampled_point_mask
    def grow(self, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        if not hasattr(self,"voxel_size"):
            self.voxel_size = 8  
        if not hasattr(self,"density_threshold"):
            self.density_threshold = density_threshold
        if not hasattr(self,"displacement_scale"):
            self.displacement_scale = displacement_scale
        flag = False
        point_cloud = self.get_xyz.detach().cpu()
        point_downsample = point_cloud.detach()
        downsampled_point_index = self.downsample_point(point_downsample)


        _, low_density_points, new_points, low_density_index = addpoint(point_cloud[downsampled_point_index],density_threshold=self.density_threshold,displacement_scale=self.displacement_scale,iter_pass=0)
        if new_points.shape[0] < 100 :
            self.density_threshold /= 2
            self.displacement_scale /= 2
            print("reduce diplacement_scale to: ",self.displacement_scale)

        elif new_points.shape[0] == 0:
            print("no point added")
            return
        global_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool)

        global_mask[downsampled_point_index] = low_density_index
        global_mask
        selected_xyz, new_xyz = self.add_point_by_mask(global_mask.to(self.get_xyz.device), self.displacement_scale)
        print("point growing,add point num:",global_mask.sum())
        if model_path is not None and iteration is not None:
            point = combine_pointcloud(point_cloud, selected_xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy())
            write_path = os.path.join(model_path,"add_point_cloud")
            os.makedirs(write_path,exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(write_path,f"iteration_{stage}{iteration}.ply"),point)
        return
    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def densify(self, max_grad, min_opacity, extent, median_dist=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        if median_dist != None:
            self.densify_and_split_dynamic(grads, max_grad, extent, median_dist=median_dist)


    def densify_dynamic(self, max_grad, min_opacity, extent, median_dist):
        # 1) call self.densify_and_clone but with a small fixed std for dynamic points
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # 2) call self.densify_and_split but override the std to e.g. displacement_scale_small
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_dynamic(grads, max_grad, extent, median_dist=median_dist)

    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)

    def compute_local_spacetime_tv(self) -> torch.Tensor:
        """
        Compute a “local” total‐variation penalty on the XT, YT, and ZT K‐planes,
        but only for those rows corresponding to dynamic Gaussians.

        Steps:
        1) Read the axis‐aligned bounding box (aabb) and grid resolution from the underlying K‐Planes:
            aabb = self._deformation.deformation_net.grid.aabb  # shape [2,3]
            └─ aabb[0] = [x_max, y_max, z_max]
            └─ aabb[1] = [x_min, y_min, z_min]
            grid_cfg = self._deformation.deformation_net.grid.grid_config[0]
            (res_x, res_y, res_z, res_t) = grid_cfg['resolution']
        2) Gather canonical (x,y,z) of only the dynamic Gaussians: xyz_dyn = self._xyz[self._dynamic_xyz].
        3) Quantize each dynamic‐Gaussian coordinate into integer plane‐rows:
            i_x = round(((x_i − x_min) / (x_max − x_min)) * (res_x − 1)), clamped to [0..res_x−1].
            Similarly for i_y, i_z.
        4) Iterate over every multi‐resolution level in self._deformation.deformation_net.grid.grids:
            • If len(grids) < 6, skip (no time‐planes here).
            • Otherwise, planes[2] is XT of shape [1, C, res_x, res_t];
                            planes[4] is YT of shape [1, C, res_y, res_t];
                            planes[5] is ZT of shape [1, C, res_z, res_t].
            For each unique i_x among dynamic Gaussians, extract plane_XT[:, i_x, :] → [C, res_t],
            compute (row[:, t] − row[:, t−1])² summed over t and channels, add to total.
            Repeat for YT (rows i_y) and ZT (rows i_z).
        5) Return total TV (a single scalar tensor).
        """
        device = self._xyz.device

        # 1) Read AABB and resolution:
        #    aabb = [[x_max, y_max, z_max],
        #            [x_min, y_min, z_min]]
        aabb = self._deformation.deformation_net.grid.aabb  # Parameter of shape [2,3]
        x_max, y_max, z_max = aabb[0]
        x_min, y_min, z_min = aabb[1]

        grid_cfg_list = self._deformation.deformation_net.grid.grid_config
        if not grid_cfg_list:
            raise RuntimeError("No grid_config found in underlying deformation network.")
        grid_cfg = grid_cfg_list[0]
        res_x, res_y, res_z, res_t = grid_cfg['resolution']

        # 2) Which Gaussians are dynamic?
        dyn_mask = self._dynamic_xyz  # bool mask shape [N_points]
        if dyn_mask.numel() == 0 or dyn_mask.sum() == 0:
            # No dynamic Gaussians → zero TV
            return torch.tensor(0.0, device=device)

        # 3) Canonical centers of dynamic Gaussians:
        xyz_dyn = self._xyz[dyn_mask]  # [K, 3]
        xs = xyz_dyn[:, 0]
        ys = xyz_dyn[:, 1]
        zs = xyz_dyn[:, 2]

        # Quantization helper: map coordinate ∈ [coord_min..coord_max] → integer index [0..res−1]
        def quantize(coord: torch.Tensor, c_min: float, c_max: float, res: int) -> torch.LongTensor:
            # Normalize to [0..1]: (coord − c_min)/(c_max − c_min)
            denom = (c_max - c_min)
            if denom == 0:
                # Degenerate bounding box; all coords equal → map all to 0
                normalized = torch.zeros_like(coord)
            else:
                normalized = (coord - c_min) / denom
            # Scale to [0..res−1], round, clamp
            scaled = normalized * (res - 1)
            idx = scaled.round().long()
            return torch.clamp(idx, 0, res - 1)

        i_x = quantize(xs, x_min, x_max, res_x)  # [K]
        i_y = quantize(ys, y_min, y_max, res_y)  # [K]
        i_z = quantize(zs, z_min, z_max, res_z)  # [K]

        total_tv = torch.tensor(0.0, device=device)

        # 4) Iterate over each multiresolution level in the K‐Planes:
        multi_res_grids = self._deformation.deformation_net.grid.grids
        for grids in multi_res_grids:
            if len(grids) < 6:
                # fewer than 6 planes → no XT/YT/ZT here
                continue

            # planes[2] = XT plane: shape [1, C, res_x, res_t]
            # planes[4] = YT plane: shape [1, C, res_y, res_t]
            # planes[5] = ZT plane: shape [1, C, res_z, res_t]
            plane_XT = grids[2].squeeze(0)  # [C, res_x, res_t]
            plane_YT = grids[4].squeeze(0)  # [C, res_y, res_t]
            plane_ZT = grids[5].squeeze(0)  # [C, res_z, res_t]

            # XT: for each unique i_x among dynamic Gaussians:
            for row_x in i_x.unique():
                row_feats = plane_XT[:, row_x, :]       # [C, res_t]
                diffs_t   = row_feats[:, 1:] - row_feats[:, :-1]  # [C, res_t−1]
                total_tv = total_tv + diffs_t.pow(2).sum()

            # YT: for each unique i_y:
            for row_y in i_y.unique():
                row_feats = plane_YT[:, row_y, :]       # [C, res_t]
                diffs_t   = row_feats[:, 1:] - row_feats[:, :-1]  # [C, res_t−1]
                total_tv = total_tv + diffs_t.pow(2).sum()

            # ZT: for each unique i_z:
            for row_z in i_z.unique():
                row_feats = plane_ZT[:, row_z, :]       # [C, res_t]
                diffs_t   = row_feats[:, 1:] - row_feats[:, :-1]  # [C, res_t−1]
                total_tv = total_tv + diffs_t.pow(2).sum()

        return total_tv


    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

    def capture_reference_deformation_by_time(self, time):
        means3D = self.get_xyz
        opacity = self._opacity
        shs = self.get_features
        scales = self._scaling
        rotations = self._rotation
        time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0],1)
        means3D_final, _, _, _, _ = self._deformation(means3D, scales,rotations, opacity, shs,time)
        self.last_seen_means_scale_rot_opacity_shs["means3D_final"] = means3D_final.detach()
        #self.last_seen_means_scale_rot_opacity_shs["scales_final"] = scales.detach()
        #self.last_seen_means_scale_rot_opacity_shs["rotations_final"] = rot.detach()
        #self.last_seen_means_scale_rot_opacity_shs["opacity_final"] = opacity.detach()
        #self.last_seen_means_scale_rot_opacity_shs["shs_final"] = shs.detach()

    def capture_visible_positions(self, positions, visibility_filter):
        self.previous_positions = positions.detach()
        self.previous_visibility = visibility_filter.detach()
    
    def compute_masked_absolute_differences(self, pcd, scales, rot, opacity, shs, mask):
        loss = torch.nanmean(abs(self.last_seen_means_scale_rot_opacity_shs["means3D_final"][mask] - pcd[mask]))
        if torch.isnan(loss).any():
            print("nan out of frame loss escaped")
            return 0
        else:
            return loss
        #abs(self.last_seen_means_scale_rot_opacity_shs["scales_final"][mask] - scales[mask]).mean() + \
        #abs(self.last_seen_means_scale_rot_opacity_shs["rotations_final"][mask] - rot[mask]).mean() + \
        #abs(self.last_seen_means_scale_rot_opacity_shs["opacity_final"][mask] - opacity[mask]).mean() + \
        #abs(self.last_seen_means_scale_rot_opacity_shs["shs_final"][mask] - shs[mask]).mean()
    