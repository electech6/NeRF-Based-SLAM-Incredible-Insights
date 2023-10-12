# package imports
import torch
import torch.nn as nn

# Local imports
from .encodings import get_encoder
from .decoder import ColorSDFNet, ColorSDFNet_v2
from .utils import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss

class JointEncoding(nn.Module):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.get_resolution()       # 从config中获得每一个体素的深度和颜色分辨率

        # ********************* 1.1 编码 *********************
        """
        首先,从config中获得的联合编码方案: 1. parametric encoding用HashGrid 2. coordinate encoding用OneBlob
        然后,通过tiny-cuda-nn实现任意方案的编码网络. 参考https://github.com/NVlabs/tiny-cuda-nn/blob/master/src/encoding.cu
        """
        self.get_encoding(config)   

        # ********************* 1.2 解码 *********************
        """
        首先,从config中获得解码网络的各个参数
        然后,为颜色color和深度sdf各创建一个2层的MLP网络,激活函数都是ReLU
        color: 最后输出维度为3, 即预测的RGB值
        sdf:   最后输出维度为16, 即预测的SDF值(1维) + 特征向量h值(15维)
        """
        self.get_decoder(config)    
        

    def get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()
        if self.config['grid']['voxel_sdf'] > 10:
            self.resolution_sdf = self.config['grid']['voxel_sdf']
        else:
            self.resolution_sdf = int(dim_max / self.config['grid']['voxel_sdf'])
        
        if self.config['grid']['voxel_color'] > 10:
            self.resolution_color = self.config['grid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['grid']['voxel_color'])
        
        print('SDF resolution:', self.resolution_sdf)

    def get_encoding(self, config):
        '''
        Get the encoding of the scene representation
        '''
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'])

        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf)

        # Sparse parametric encoding (Color)
        if not self.config['grid']['oneGrid']:
            print('Color resolution:', self.resolution_color)
            self.embed_fn_color, self.input_ch_color = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_color)

    def get_decoder(self, config):
        '''
        Get the decoder of the scene representation
        '''
        if not self.config['grid']['oneGrid']:
            self.decoder = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            self.decoder = ColorSDFNet_v2(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        
        # 是否批处理。这里设为None，就是没有批处理
        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)

    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        # 论文公式5
        weights = torch.sigmoid(sdf / args['training']['trunc']) * torch.sigmoid(-sdf / args['training']['trunc'])  # [2048,85]

        signs = sdf[:, 1:] * sdf[:, :-1]    # [2048,84] 相邻2个SDF之间是否发生符号变化
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))    # 找到SDF符号发生变化的部分。mask设为1，否则设为0  [2048,84]
        inds = torch.argmax(mask, axis=1)   # 在每一行查找最大值的索引  [2048]
        inds = inds[..., None]              # [2048,1]
        z_min = torch.gather(z_vals, 1, inds) # The first surface [2048,1] 获取第一个表面的z值
        mask = torch.where(z_vals < z_min + args['data']['sc_factor'] * args['training']['trunc'], torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
    def raw2outputs(self, raw, z_vals, white_bkgd=False):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # raw光线的前三列是 rgb值 [N_rays, N_samples, 3]  [2048,85,3]
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)   # 通过raw的第四列深度值sdf,得到论文公式5的权重  [2048,85]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # 论文4式，计算rgb图 [N_rays, 3] [2048,3]

        depth_map = torch.sum(weights * z_vals, -1)    # 论文4式，计算深度图    [2048]
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1) # 深度方差,表示每条射线深度的变化情况
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) # 视差图disparity map,相机到物体的距离的倒数,表达物体的远近
        acc_map = torch.sum(weights, -1)    # 累积权重图，表示沿每条射线的累积权重

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var
    
    def query_sdf(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat
    
    def query_color(self, query_points):
        return torch.sigmoid(self.query_color_sdf(query_points)[..., :3])
    
    def query_color_sdf(self, query_points):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        # ********************* encoder *********************
        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)

        # ********************* decoder 得到颜色和深度 *********************
        if not self.config['grid']['oneGrid']:
            embed_color = self.embed_fn_color(inputs_flat)
            return self.decoder(embed, embe_pos, embed_color)
        return self.decoder(embed, embe_pos)
    
    def run_network(self, inputs):
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [174080,3] <=== [2048,85,3]
        
        # Normalize the input to [0, 1] (TCNN convention)归一化处理 2048根射线,每根射线85个采样点到[0,1]
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        
        # 将inputs_flat分成较小的批次，然后对每个批次使用query_color_sdf函数(也就是encoder+decoder网络)
        # 得到每个点的深度和颜色信息
        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)    # [174080,4] 4:深度1维,颜色3维
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])   # [2048, 85, 4]

        return outputs
    
    def render_surface_color(self, rays_o, normal):
        '''
        Render the surface color of the points.
        Params:
            points: [N_rays, 1, 3]
            normal: [N_rays, 3]
        '''
        n_rays = rays_o.shape[0]
        trunc = self.config['training']['trunc']
        z_vals = torch.linspace(-trunc, trunc, steps=self.config['training']['n_range_d']).to(rays_o)
        z_vals = z_vals.repeat(n_rays, 1)
        # Run rendering pipeline
        
        pts = rays_o[...,:] + normal[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        rgb, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])
        return rgb
    
    def render_rays(self, rays_o, rays_d, target_d=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]    # 光线的个数2048

        # ! -------------------- 2.1 Ray samping -------------------- 
        # ********************* 在光线上(深度)取样，确保取样点深度都是正的 *********************
        if target_d is not None:
            # 在深度-range_d到range_d=25范围内生成n_range_d=21个等间隔张量，并转换到与target_d张量相同的设备和数据类型上
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
            # 将[1,21]的张量z_samples[None, :]扩展为[2048,1]的张量，并加上[2048,1]的target_d
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            # 将目标深度为负的那些取样点改为在深度near=0到far=5的范围内生成n_range_d==21个等间隔张量
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

            if self.config['training']['n_samples_d'] > 0:
                # 再在深度near=0到far=5的范围内生成n_samples_d=64个取样点
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples_d'])[None, :].repeat(n_rays, 1).to(rays_o)
                # 将[2048,64]的z_vals和[2048,21]的z_samples合并，然后按照深度值进行排序。这将确保深度值是有序的
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # [2048,85]
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # ********************* 给取样点的深度添加扰动 *********************
        # Perturb sampling depths 
        if self.config['training']['perturb'] > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])      # [2048,84]
            upper = torch.cat([mids, z_vals[...,-1:]], -1)      # [2048,85]
            lower = torch.cat([z_vals[...,:1], mids], -1)       # [2048,85]
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)  

        # ********************* 通过encoding和decoder网络,计算深度和颜色，得到图 *********************
        # Run rendering pipeline
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # 每条射线取样点的坐标 [2048,85,3] [N_rays, N_samples, 3]
        raw = self.run_network(pts)     # 使用encoding和decoder网络,计算得到每根光线的每个采样点的深度和颜色    [2048, 85, 4]
        
        # 将上面算出的每个采样点的深度和颜色原始信息,转换为rgb图，视差图，累积权重图，权重，深度图，深度方差
        rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # ********************* 重要点采样 *********************
        # ? 论文Depth-guided Sampling处说Importance sampling没什么效果,还降低slam速度,所以不用
        if self.config['training']['n_importance'] > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = rgb_map, disp_map, acc_map, depth_map, depth_var

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            raw = self.run_network(pts)
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # ********************* 记录返回值 *********************
        # Return rendering outputs: rgb图，深度图，视差图，累积权重图，深度方差, 所有采样点的深度
        ret = {'rgb' : rgb_map, 'depth' :depth_map, 
               'disp_map' : disp_map, 'acc_map' : acc_map, 
               'depth_var':depth_var,}
        ret = {**ret, 'z_vals': z_vals}

        ret['raw'] = raw

        if self.config['training']['n_importance'] > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['depth_var0'] = depth_var_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        return ret
    
    def forward(self, rays_o, rays_d, target_rgb, target_d, global_step=0):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # ********************* 获得rgb图，深度图，视差图，累积权重图，深度方差, 所有采样点的深度 *********************
        # Get render results
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d)

        if not self.training:
            return rend_dict
        
        # Get depth and rgb weights for loss
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # ********************* 根据论文6式计算颜色和深度的损失函数 *********************
        # Get render loss
        rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
        psnr = mse2psnr(rgb_loss)
        depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

        if 'rgb0' in rend_dict:
            rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
            depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # ********************* 根据论文7式和8式计算sdf和free-space损失 *********************
        # Get sdf loss
        z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = rend_dict['raw'][..., -1]  # [N_rand, N_samples + N_importance]
        truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']
        fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, truncation, 'l2', grad=None)         
        

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
        }

        return ret
