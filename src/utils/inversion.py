import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy
from lpips import LPIPS
from tqdm import tqdm
from gghead_jittor.face_utils import FLAME, flame_postprocess, _cached_flame
from gghead_jittor.face_utils.deca_onnx import FLAMERecon, CropFace, angle2cam
from gghead_jittor.face_utils.eg3d import crop_image
from gghead_jittor.face_utils.bgrm import modnet, mask_image
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose
from gghead_jittor.constants import DEFAULT_INTRINSICS
from eg3d.datamanager.nersemble import encode_camera_params, decode_camera_params
from eg3d import dnnlib
from gghead_jittor.env import REPO_ROOT_DIR
from gaussian_splatting.utils.loss_utils import ssim
from skimage import transform
from kornia.geometry.transform import warp_affine
from src.utils.perceptual_utils import FacePerceptualLoss
from gghead_jittor.model_manager.finder import find_model_manager
from gghead_jittor.models.gghead_model import GGHeadModel
from src.utils.camera_utils import get_c3,  FaceRecon

from gaussian_splatting.utils.sh_utils import C0, eval_sh
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.arguments import PipelineParams2
from dreifus.matrix import Pose, Intrinsics
flame_recon = FLAMERecon()
torch_hub_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch_hub')
os.environ['TORCH_HOME'] = torch_hub_dir
#==================== PTI
class PTI:
    def __init__(self, G, device, pti_steps=350, inv_steps=450, log_images_step=10,percep_lambda=0.04,
                 l2_lambda=1, lpips_lambda=1, lpips_threshold=0.01,lr_pti=1e-2,lr_inv=1e-2,
                 use_ball_holder=True, ball_reg_iterval=10):
        self.w_pivots = {}
        self.image_counter = 0
        self.G = G.eval()
        self.state_dict = self.G.state_dict()
        # self.new_G = None
        self.device = device
        self.pti_steps=pti_steps
        self.log_images_step=log_images_step
        self.lr_pti = lr_pti
        self.lr_inv = lr_inv
        self.l2_lambda = l2_lambda
        self.lpips_lambda = lpips_lambda
        self.lpips_threshold = lpips_threshold
        self.use_ball_holder = use_ball_holder
        self.ball_reg_iterval=ball_reg_iterval
        if self.lpips_lambda > 0 or self.use_ball_holder or self.lpips_threshold > 0:
            self.lpips_loss = LPIPS(net='alex', verbose=False).to(self.device).eval()
        self.restart_training()
        self.percep_lambda=percep_lambda
        self.projector = Projector(self.G, device=device, num_iter=inv_steps,log_images_step=log_images_step,percep_lambda=self.percep_lambda,lr=self.lr_inv)
        self.percep_loss = FacePerceptualLoss(loss_type='l1', weighted=True).to(self.device)
    def restart_training(self):
        if not hasattr(self, 'new_G'):
            self.new_G = GGHeadModel(self.G._config).to(self.device)
        # self.new_G = copy.deepcopy(self.G)
        
        self.new_G.load_state_dict(self.state_dict, strict=False)
        for p in self.new_G.parameters():
            p.requires_grad = True
        self.new_G.gghead_updata()
        self.new_G._uv_idx = self.new_G._uv_idx.to(self.device)
        if self.use_ball_holder:
            self.space_regulizer = SpaceRegulizer(self.G, self.lpips_loss)
        self.optimizer = torch.optim.Adam(self.new_G.parameters(), lr=self.lr_pti)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.pti_steps, eta_min=0.0)  
    def calc_inversions(self, image, verbose=False,cs=None,optim_w=False,**kwargs):
        result = self.projector(image, verbose=verbose,cs=cs,optim_w=optim_w,**kwargs)
        return result
    
    def calc_loss(self, generated_images, real_images, new_G, w_batch, mapping_params={}, synthesis_params={}, use_ball_holder=True):
        loss = 0.0
        
        if self.l2_lambda > 0:
            l2_loss = F.mse_loss(generated_images, real_images)
            loss += l2_loss * self.l2_lambda
        
        if self.lpips_lambda > 0 or self.lpips_threshold > 0:
            lpips_loss = self.lpips_loss(generated_images, real_images)
            lpips_loss = torch.mean(torch.squeeze(lpips_loss))
            loss += lpips_loss * self.lpips_lambda
        if self.percep_lambda > 0 :
            loss += self.percep_loss(generated_images, real_images.to(generated_images),normalize=True)* self.percep_lambda
        if use_ball_holder:
            loss += self.space_regulizer.space_regulizer_loss(new_G, w_batch, mapping_params=mapping_params, synthesis_params=synthesis_params)
    
        return loss, lpips_loss if self.lpips_threshold > 0 else None
    
    def train(self, image, verbose=False,cs=None,optim_w=False,**kwargs):
        self.restart_training()
        result = self.calc_inversions(image, verbose=verbose,cs=cs,optim_w=optim_w,**kwargs)
        real_images = result['target']
        real_images = real_images.div(255/2).sub(1).clamp(-1.0+1e-8, 1.0-1e-8)
        w_pivot = result['w']
        synthesis_params = result['synthesis_params']
        mapping_params = result['mapping_params']
        
        tbar = tqdm(range(self.pti_steps), ncols=80, disable=not verbose, desc="[PTI]Training")
        for i in tbar:
            ret = self.forward(w_pivot, synthesis_params=synthesis_params)
            generated_images = ret['image'].clamp(-1.0+1e-8, 1.0-1e-8)
            
            loss, lpips_loss = self.calc_loss(generated_images, real_images, self.new_G, w_pivot,
                                  synthesis_params=synthesis_params,
                                  mapping_params=mapping_params,
                                  use_ball_holder=(self.use_ball_holder and i % self.ball_reg_iterval == 0))
            msg = f"[PTI]Loss: {loss.item():.3f}"
            if lpips_loss is not None:
                msg += f"; LPIPS: {lpips_loss.item():.3f}"
            tbar.set_description(msg)
            if lpips_loss is not None and lpips_loss < self.lpips_threshold:
                break
          
              
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            # if i % self.log_images_step == 0:
            #     result['images'].append(generated_images.detach().cpu()[0])
        with torch.no_grad():
            c_3 = get_c3(result['synthesis_params']['c'],0.3,0.1,self.device)
            ret = self.forward(w_pivot, synthesis_params=synthesis_params)
            generated_images = ret['image'].clamp(-1.0+1e-8, 1.0-1e-8)
            result['image'] = generated_images.detach().cpu()
            
            synthesis_params['c'] = c_3[1]
            ret = self.forward(w_pivot, synthesis_params=synthesis_params)
            generated_images = ret['image'].clamp(-1.0+1e-8, 1.0-1e-8)
            result['image1'] = generated_images.detach().cpu()

            synthesis_params['c'] = c_3[2]
            ret = self.forward(w_pivot, synthesis_params=synthesis_params)
            generated_images = ret['image'].clamp(-1.0+1e-8, 1.0-1e-8)
            result['image2'] = generated_images.detach().cpu()
            result['f_image'] = real_images.detach().cpu()
        return result
        
    def forward(self, w, synthesis_params={}):
        
        return self.new_G.synthesis(w, noise_mode='const', **synthesis_params)

#==================== W+ Projector

class Projector:
    def __init__(self, G, device, w_avg=None, w_std=None, num_iter=100, log_images_step=10,lr=0.01,l2_lambda=1, lpips_lambda=1,percep_lambda=1, 
                 sketch_gen=None,lpips_threshold=0.06,face_recon_model=None):
        self.G = G
        self.device = device
        self.w_avg = w_avg
        self.w_std = w_std
        self.num_iter = num_iter
        self.log_images_step = log_images_step
        self.l2_lambda=l2_lambda
        self.lpips_lambda=lpips_lambda
        self.percep_lambda = percep_lambda
        self.lpips_threshold=lpips_threshold
        # Prepare hyperparameters
        self.initial_learning_rate=lr
        self.initial_noise_factor=0.05
        self.lr_rampdown_length=0.25
        self.lr_rampup_length=0.05
        self.noise_ramp_length=0.75
        self.batch_size = 1
        self.noise_reg_loss_weight = 1e5
        if self.lpips_lambda > 0  or self.lpips_threshold > 0:
            self.lpips_loss = LPIPS(net='alex', verbose=False).to(self.device).eval()

        # Prepare common parameters
        c_front = encode_camera_params(
            Pose(matrix_or_rotation=np.eye(3), translation=(0, 0., 2.7), pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)
        self.c_front = torch.from_numpy(c_front).to(self.device).unsqueeze(0).to(self.device)
        # if self.w_avg is None or self.w_std is None:
        #     ws = []
        #     z = torch.zeros(1, self.G._config.z_dim, device=self.device).to(self.device)
        #     ws = self.G.mapping(z, self.c_front.repeat(1, 1))
        #     self.w_avg = ws[:, :1, :].mean(dim=0, keepdim=True)
        #     self.w_std = ws[:, :1, :].std(dim=0, keepdim=True)
        #     self.z = z
        #     self.z_std = torch.ones(1, self.G._config.z_dim, device=self.device)
        # Load VGG16 feature detector.
        if self.w_avg is None or self.w_std is None:
            ws = []
            z = torch.randn(10000, self.G._config.z_dim, device=self.device)
            ws = self.G.mapping(z, self.c_front.repeat(10000, 1))
            self.w_avg = ws[:, :1, :].mean(dim=0, keepdim=True)
            self.w_std = ws[:, :1, :].std(dim=0, keepdim=True)
        url = './assets/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(self.device)
        self.percep_loss = FacePerceptualLoss(loss_type='l1', weighted=True).to(self.device)
        if face_recon_model:
            self.face_recon_model = face_recon_model#FaceRecon(self.device)
        else:
            self.face_recon_model = FaceRecon(self.device)
        if sketch_gen:
            self.sketch_gen = sketch_gen#FaceRecon(self.device)
        else:
            self.sketch_gen = None
    def __call__(self, image, verbose=False,cs=None,optim_w=False,**kwargs):
        if optim_w:
            return self.project_w(image, verbose,cs=cs,**kwargs)
        else:
            return self.project(image, verbose,cs=cs,)
    def project(self, image, verbose=False,cs=None,):
        result = {}
        # if self.device in _cached_flame:
        #     flame = _cached_flame[self.device]
        # else:
        #     flame = FLAME().to(self.device)
        #     _cached_flame[self.device] = flame
        
        # code_dict = flame_recon.reconstruct_FLAME_from_rawimg(image)
        # code_dict = {
        #     "shape": torch.tensor(code_dict['shape'], device=self.device, dtype=torch.float32).reshape(1, -1),
        #     "exp": torch.tensor(code_dict['exp'], device=self.device, dtype=torch.float32).reshape(1, -1),
        #     "pose": torch.tensor(code_dict['pose'], device=self.device, dtype=torch.float32).reshape(1, -1),
        # }
        # verts = flame(code_dict['shape'], code_dict['exp'], code_dict['pose'])[0]
        # verts = flame_postprocess(verts, flame.faces_tensor)[0]

        # image = crop_image(image)
        mask = modnet(image[None])[0]
        image = mask_image(image, mask)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        result['target'] = image
        real_images = image/255*2-1
        image = torch.nn.functional.interpolate(image, (256, 256), mode='bilinear')
        with torch.no_grad():
            target_features = self.vgg16(image, resize_images=False, return_lpips=True) #?

        
        _, c = self.face_recon_model((real_images+1)/2)
        # c[0,[3,7,11]] = c[0,[3,7,11]]-torch.tensor([0,0.006,0.161]).to(c)
        # print((((c[0,[3,7,11]]-torch.tensor([0,0.006,0.161]).to(c))**2).sum())**(0.5)-2.7)
        # import pdb
        # pdb.set_trace()
        if cs!=None:
            c = cs
        # c = encode_camera_params(
        #     Pose(matrix_or_rotation=code_dict['pose'][:, :3].cpu().numpy()[0], translation=(0, 0, 3.5), pose_type=PoseType.CAM_2_WORLD,
        #          camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)
        # c = torch.from_numpy(c).cuda().unsqueeze(0)
        # code_dict['pose'][:, :3] *= 0

        ws = self.w_avg
        
        start_w = ws.clone().repeat(1, self.G.backbone.mapping.num_ws, 1)
        w_opt = torch.nn.Parameter(start_w)
        
        # Setup noise inputs.
        noise_bufs = {name: buf for (name, buf) in self.G.named_buffers() if 'noise_const' in name and 'synthesis' in name}
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True


        z = self.z.clone() 
        z_opt = torch.nn.Parameter(z)

        # optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
        #                             lr=self.initial_learning_rate,)
        optimizer = torch.optim.Adam([z_opt] , betas=(0.9, 0.999),
                                    lr=self.initial_learning_rate,)            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_iter, eta_min=0.0)               

        c = c.repeat(self.batch_size, 1)
        # verts = verts.repeat(self.batch_size, 1, 1)
        # exp = code_dict["exp"].repeat(self.batch_size, 1)
        # pose = code_dict["pose"].repeat(self.batch_size, 1)
        synthesis_params = {
            'c': c,
        }
        mapping_params = {
            'c': self.c_front,
        }
        result['mapping_params'] = mapping_params
        result['synthesis_params'] = synthesis_params
        image = image.repeat(self.batch_size, 1, 1, 1)
        images = []
        tbar = tqdm(range(self.num_iter), ncols=80, disable=not verbose, desc="[Inv]Optimizing")
        for i in tbar:
            # Noise and learning rate schedule.
            t = i / self.num_iter
            z_noise_scale = self.z_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            z_noise = torch.randn_like(z_opt.repeat(self.batch_size, 1)) * z_noise_scale
            ws = self.G.mapping(z_opt+z_noise, self.c_front.repeat(1, 1))
            # ws = (ws + w_noise)
            ret = self.G.synthesis(ws, noise_mode='const', **synthesis_params)
            pred_img = ret['image'] 
            
            # Calculate loss.
            loss = 0.
            msg = "[Inv]"
            generated_images=(pred_img.clone())
            pred_img = pred_img.add(1).mul(255/2)
            pred_img = torch.nn.functional.interpolate(pred_img, (256, 256), mode='bilinear')
            pred_features = self.vgg16(pred_img, resize_images=False, return_lpips=True)
            dist = (pred_features - target_features).square().sum()
            loss += dist
            loss_, lpips_loss = self.calc_loss(generated_images, real_images, self.G, w_opt,
                                  synthesis_params=synthesis_params,
                                  mapping_params=mapping_params,
                                  use_ball_holder=False)
            loss +=loss_
            msg += f"[PTI]Loss_: {loss_.item():.3f}"
            if lpips_loss is not None:
                msg += f"; LPIPS: {lpips_loss.item():.3f}"
            msg += f"Dist:{dist.item():.3f}; "
            noise_reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]
                while True:
                    noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            msg += f"NoiseReg:{noise_reg_loss.item():.3f}; "
            loss += noise_reg_loss * self.noise_reg_loss_weight
            msg += f"Total:{loss.item():.3f}; "
            tbar.set_description(msg)

            # Update
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if i % self.log_images_step == 0:
                images.append(ret['image'].detach().cpu()[0])
        result['w'] = ws.detach()
        result['image'] = ret['image'].detach().cpu()[0]
        result['images'] = images
        result['c'] = c
        result['z'] = z_opt
        result['c_front'] = self.c_front
        # result['image_target'] = image
        return result
    
    def project_gaussian(self, image, verbose=False,cs=None,w_start=None,conditions=None):
        result = {}
        # if self.device in _cached_flame:
        #     flame = _cached_flame[self.device]
        # else:
        #     flame = FLAME().to(self.device)
        #     _cached_flame[self.device] = flame
        
        # code_dict = flame_recon.reconstruct_FLAME_from_rawimg(image)
        # code_dict = {
        #     "shape": torch.tensor(code_dict['shape'], device=self.device, dtype=torch.float32).reshape(1, -1),
        #     "exp": torch.tensor(code_dict['exp'], device=self.device, dtype=torch.float32).reshape(1, -1),
        #     "pose": torch.tensor(code_dict['pose'], device=self.device, dtype=torch.float32).reshape(1, -1),
        # }
        # verts = flame(code_dict['shape'], code_dict['exp'], code_dict['pose'])[0]
        # verts = flame_postprocess(verts, flame.faces_tensor)[0]

        # image = crop_image(image)
        # breakpoint()
        mask = modnet(image[None])[0]
        image = mask_image(image, mask)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        result['target'] = image
        real_images = image/255*2-1
        image = torch.nn.functional.interpolate(image, (256, 256), mode='bilinear')
        with torch.no_grad():
            target_features = self.vgg16(image, resize_images=False, return_lpips=True) #?

        
        _, c = self.face_recon_model((real_images+1)/2)
        # c[0,[3,7,11]] = c[0,[3,7,11]]-torch.tensor([0,0.006,0.161]).to(c)
        # print((((c[0,[3,7,11]]-torch.tensor([0,0.006,0.161]).to(c))**2).sum())**(0.5)-2.7)
        # import pdb
        # pdb.set_trace()
        if cs!=None:
            c = cs
        # c = encode_camera_params(
        #     Pose(matrix_or_rotation=code_dict['pose'][:, :3].cpu().numpy()[0], translation=(0, 0, 3.5), pose_type=PoseType.CAM_2_WORLD,
        #          camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)
        # c = torch.from_numpy(c).cuda().unsqueeze(0)
        # code_dict['pose'][:, :3] *= 0

        # ws = self.w_avg
        w_begin = w_start.detach().clone()
        w_start = w_start.detach()
        # start_w1 = w_start[:,:14].clone()
        # start_w2 = w_start[:,14:].clone()#.repeat(1, self.G.backbone.mapping.num_ws, 1)
        # start_w = torch.nn.Parameter(w_start.clone())
        # w_opt = torch.cat([start_w1,start_w2],1)
        # Setup noise inputs.
        noise_bufs = {name: buf for (name, buf) in self.G.named_buffers() if 'noise_const' in name and 'synthesis' in name}
        for buf in noise_bufs.values():
            buf = buf.detach()
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True


        # z = self.z.clone() 
        # z_opt = torch.nn.Parameter(z)

        # optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
        #                             lr=self.initial_learning_rate,)
        part1 = w_start[:, :10]

# 2. 取出第二维度的第14个及之后的元素 (索引 14-19)
        part2 = w_start[:, 14:]

        # 3. 沿着第二个维度 (dim=1) 将它们拼接起来
        # result = torch.cat((part1, part2), dim=1)
        part1 = torch.nn.Parameter(part1.clone())
        part2 = torch.nn.Parameter(part2.clone())
        for i,condition in enumerate(conditions):
            conditions[i] = torch.nn.Parameter(condition.clone())
        # optimizer = torch.optim.Adam([part1,part2] , betas=(0.9, 0.999),
        #                             lr=self.initial_learning_rate,)       
        # optimizer1 = torch.optim.Adam(
        #     [
        #         {'params': part1, 'lr': self.initial_learning_rate},
        #         {'params': part2, 'lr': 0.1*self.initial_learning_rate}
                
        #     ], 
        #     betas=(0.9, 0.999),
        #       # 这个 lr 现在作为未在组中指定 lr 的参数的默认值
        # )      
        
        optimizer1 = torch.optim.Adam(
           [ {'params': part1, 'lr': 1e-2},
        {'params': part2, 'lr': 1e-3}] ,
            betas=(0.9, 0.999),
            #  lr=1e-3#self.initial_learning_rate, # 这个 lr 现在作为未在组中指定 lr 的参数的默认值
        )     
        optimizer2 = torch.optim.Adam(
            [
                
                {'params': part2, 'lr': 0.1*self.initial_learning_rate}
            ], 
            betas=(0.9, 0.999),
              # 这个 lr 现在作为未在组中指定 lr 的参数的默认值
        )     
        # start_w = torch.cat((part1,w_start[:, 10:14], part2,w_start[:, [20]]), dim=1)
        # start_w = torch.cat((part1,w_start[:, 10:14], part2), dim=1)       
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=self.num_iter, eta_min=0.0)               
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=self.num_iter, eta_min=0.0) 
        c = c.repeat(self.batch_size, 1)
        # verts = verts.repeat(self.batch_size, 1, 1)
        # exp = code_dict["exp"].repeat(self.batch_size, 1)
        # pose = code_dict["pose"].repeat(self.batch_size, 1)
        synthesis_params = {
            'c': c,
        }
        mapping_params = {
            'c': self.c_front,
        }
        result['mapping_params'] = mapping_params
        result['synthesis_params'] = synthesis_params
        image = image.repeat(self.batch_size, 1, 1, 1)
        images = []
        tbar = tqdm(range(self.num_iter), ncols=80, disable=not verbose, desc="[Inv]Optimizing")
        for i in tbar:
            # Noise and learning rate schedule.
            # Noise and learning rate schedule.
            t = i / self.num_iter
            w_noise_scale = self.w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            # z_noise_scale = self.z_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            # Synth images from opt_w.
            # z_noise = torch.randn_like(z_opt.repeat(self.batch_size, 1)) * z_noise_scale
            # ws = self.G.mapping(z_opt+z_noise, self.c_front.repeat(1, 1))
            w_noise = torch.randn((self.batch_size, 21, 1)).to(self.device) * w_noise_scale
            # if w_start!=None:
            # ws = w_opt
            # else:
            # w_noise[:,:14] = start_w1# + w_noise[:,:14])
            # w_noise[:,14:] = start_w2+ w_noise[:,14:]
            # ws = start_w + w_noise
            # ws = w_noise + torch.cat((part1,w_start[:, 10:14], part2,w_start[:, [20]]), dim=1)
            ws = w_noise + torch.cat((part1, w_start[:,10:14],part2,), dim=1)
            # ws = w_start
            ws[:, 10:14] = w_start[:, 10:14]
            # ws[:, [20]] = w_start[:, [20]]
            # ws = (ws + w_noise)
            # c_front = encode_camera_params(
            #         Pose(matrix_or_rotation=np.eye(3), translation=(0, 0.0, 2.7), pose_type=PoseType.CAM_2_WORLD,
            #             camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)
            # c_front = torch.from_numpy(c_front).cuda().unsqueeze(0)
            sh_ref_cam, intrinsics = decode_camera_params(self.c_front[0].cpu())

            # ret = self.G.synthesis(ws, noise_mode='const', **synthesis_params)
            ret = self.G.synthesis(ws, c, sh_ref_cam=sh_ref_cam, noise_mode='const',
                                           neural_rendering_resolution=512,conditions=conditions)
            # pred_img_geo,features_dc,features_rest = self.gs_gen(geo=True,c=c,features_dc=None,features_rest=None)
            # pred_img_ap = self.gs_gen(geo=False,c=c,features_dc=features_dc,features_rest=features_rest)
            pred_img = ret['image'] 
#-----------------------------------------------1-----------------------------------------------------------------
            # pred_img = pred_img_ap
            # # Calculate loss.
            # loss = 0.
            # msg = "[Inv]"
            # generated_images=(pred_img.clone())
            # pred_img = pred_img.add(1).mul(255/2)
            # pred_img = torch.nn.functional.interpolate(pred_img, (256, 256), mode='bilinear')
            # pred_features = self.vgg16(pred_img, resize_images=False, return_lpips=True)
            # dist = (pred_features - target_features).square().sum()
            # loss += dist
            # loss_, lpips_loss, l2_loss,percep_loss = self.calc_loss(generated_images, real_images, self.G, ws,#sketch_loss,sketch_lpips_loss
            #                       synthesis_params=synthesis_params,
            #                       mapping_params=mapping_params,
            #                       use_ball_holder=False)
            # loss +=loss_
            # msg += f"[PTI]Loss_: {loss_.item():.3f}"
            # msg += f"[PTI]l2_loss: {l2_loss.item():.3f}"
            # msg += f"[PTI]percep_loss: {percep_loss.item():.3f}"
            # # msg += f"[PTI]sketch_loss: {sketch_loss.item():.3f}"
            # # msg += f"[PTI]sketch_lpips_loss: {sketch_lpips_loss.item():.3f}"
            # if lpips_loss is not None:
            #     msg += f"; LPIPS: {lpips_loss.item():.3f}"
            # msg += f"Dist:{dist.item():.3f}; "
            # noise_reg_loss = 0.0
            # for v in noise_bufs.values():
            #     noise = v[None, None, :, :]
            #     while True:
            #         noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
            #         noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
            #         if noise.shape[2] <= 8:
            #             break
            #         noise = F.avg_pool2d(noise, kernel_size=2)
            # msg += f"NoiseReg:{noise_reg_loss.item():.3f}; "
            # loss += noise_reg_loss * self.noise_reg_loss_weight
            # msg += f"Total:{loss.item():.3f}; "
            # tbar.set_description(msg)

            # # Update
            # loss.backward(retain_graph=True)
            # optimizer2.step()
            # scheduler2.step()
            # optimizer2.zero_grad()

#-----------------------------------------------2---------------------------------------------- 

            # pred_img = pred_img_geo
            loss = 0.
            msg = "[Inv]"
            generated_images=(pred_img.clone())
            pred_img = pred_img.add(1).mul(255/2)
            pred_img = torch.nn.functional.interpolate(pred_img, (256, 256), mode='bilinear')
            pred_features = self.vgg16(pred_img, resize_images=False, return_lpips=True)
            dist = (pred_features - target_features).square().sum()
            loss += dist
            loss_, lpips_loss, l2_loss,percep_loss = self.calc_loss(generated_images, real_images, self.G, ws,#sketch_loss,sketch_lpips_loss
                                  synthesis_params=synthesis_params,
                                  mapping_params=mapping_params,
                                  use_ball_holder=False)
            loss +=loss_
            msg += f"[PTI]Loss_: {loss_.item():.3f}"
            msg += f"[PTI]l2_loss: {l2_loss.item():.3f}"
            msg += f"[PTI]percep_loss: {percep_loss.item():.3f}"
            # msg += f"[PTI]sketch_loss: {sketch_loss.item():.3f}"
            # msg += f"[PTI]sketch_lpips_loss: {sketch_lpips_loss.item():.3f}"
            if lpips_loss is not None:
                msg += f"; LPIPS: {lpips_loss.item():.3f}"
            msg += f"Dist:{dist.item():.3f}; "
            noise_reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]
                while True:
                    noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            msg += f"NoiseReg:{noise_reg_loss.item():.3f}; "

            # loss += noise_reg_loss * self.noise_reg_loss_weight
            msg += f"Total:{loss.item():.3f}; "
            tbar.set_description(msg)

            # Update
            loss.backward()
            optimizer1.step()
            scheduler1.step()
            optimizer1.zero_grad()
            if i % self.log_images_step == 0:
                images.append(ret['image'].detach().cpu()[0])
        result['w'] = ws.detach()
        # result['w_gap'] = ws.detach() - w_begin
        result['image'] = ret['image'].detach().cpu()[0]
        result['images'] = images
        result['c'] = c
        # result['conditions'] = conditions
        # result['z'] = z_opt
        result['c_front'] = self.c_front
        # result['image_target'] = image
        return result
    def gs_gen(self,geo=False,c=None,features_dc=None,features_rest=None):
        
        gen_images = []
        c_front = encode_camera_params(
                    Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 2.7), pose_type=PoseType.CAM_2_WORLD,
                         camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)
        c_front = torch.from_numpy(c_front).cuda().unsqueeze(0)
        sh_ref_cam, intrinsics = decode_camera_params(c_front[0].cpu())
        intrinsics = intrinsics.rescale(512, inplace=False) 
        gaussian_sh_ref_cam = pose_to_rendercam(sh_ref_cam, intrinsics, 512, 512, device=self.device)
        sh_degree = self.G._config.gaussian_attribute_config.sh_degree
        n_feature_channels = self.G._config.gaussian_attribute_config.n_color_channels

        
        c2w = c[:,:16]
        cam2world_matrix = c2w.view(4, 4)
        intrinsics_matrix = DEFAULT_INTRINSICS.reshape(3, 3)
        cam_2_world_pose = Pose(
            cam2world_matrix.cpu().numpy(),
            pose_type=PoseType.CAM_2_WORLD,
            disable_rotation_check=True,
        )
        intrinsics = Intrinsics(intrinsics_matrix)
        intrinsics = intrinsics.rescale(512, inplace=False)
        gaussian_camera = pose_to_rendercam(
            cam_2_world_pose, intrinsics, 512, 512, device=self.device
        )
        # f_gen_images = []
        # if batch_size==3:
        #     print(gs_attr_dict["xyz"])
        #     pdb.set_trace()
        # for i in range(gs_attr_dict["xyz"].shape[0]):
        if not geo:
            self.G._gaussian_model._features_dc = features_dc
            self.G._gaussian_model._features_rest = features_rest

            self.G._gaussian_model._xyz = self.G._gaussian_model._xyz.detach()
            self.G._gaussian_model._scaling = self.G._gaussian_model._scaling.detach()
            self.G._gaussian_model._rotation = self.G._gaussian_model._rotation.detach()
            self.G._gaussian_model._opacity = self.G._gaussian_model._opacity.detach()
        if  geo:
            features_dc = self.G._gaussian_model._features_dc.clone()
            self.G._gaussian_model._features_dc = self.G._gaussian_model._features_dc.detach()
            features_rest = self.G._gaussian_model._features_rest.clone()
            self.G._gaussian_model._features_rest = self.G._gaussian_model._features_rest.detach()  # [G, SH-1, 3]
        
        shs_view = self.G._gaussian_model.get_features.view(-1, (sh_degree + 1) ** 2, n_feature_channels).permute(0, 2, 1)
        dir_pp = (self.G._gaussian_model.get_xyz - gaussian_sh_ref_cam.camera_center.repeat(1, 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
        sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        override_color = colors
        # self._gaussian_model._features_dc = self.color[:, [0]]
        # self._gaussian_model._features_rest = self.color[:, 1:]
        # if bg !=None:
        #     rendered_image = render(gaussian_camera, self._gaussian_model, PipelineParams2(), bg,
        #                         override_color=override_color)['render']
        # else:
        rendered_image = render(gaussian_camera, self.G._gaussian_model, PipelineParams2(), self.G._gaussian_bg_train,
                            override_color=override_color)['render']

        # gen_images.append(rendered_image)

            # f_rendered_image = render(f_cam[i], self._gaussian_model, PipelineParams2(), self._gaussian_bg_train,
            #                         override_color=None)['render']

            # f_gen_images.append(f_rendered_image)

        # gen_images = torch.stack(gen_images, 0)
        if  geo:
            return rendered_image[None]*2-1,features_dc,features_rest
        else:
            return rendered_image[None]*2-1
    def project_w(self, image, verbose=False,cs=None,w_start=None):
        result = {}
        # if self.device in _cached_flame:
        #     flame = _cached_flame[self.device]
        # else:
        #     flame = FLAME().to(self.device)
        #     _cached_flame[self.device] = flame
        
        # code_dict = flame_recon.reconstruct_FLAME_from_rawimg(image)
        # code_dict = {
        #     "shape": torch.tensor(code_dict['shape'], device=self.device, dtype=torch.float32).reshape(1, -1),
        #     "exp": torch.tensor(code_dict['exp'], device=self.device, dtype=torch.float32).reshape(1, -1),
        #     "pose": torch.tensor(code_dict['pose'], device=self.device, dtype=torch.float32).reshape(1, -1),
        # }
        # verts = flame(code_dict['shape'], code_dict['exp'], code_dict['pose'])[0]
        # verts = flame_postprocess(verts, flame.faces_tensor)[0]

        # image = crop_image(image)
        mask = modnet(image[None])[0]
        image = mask_image(image, mask)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        result['target'] = torch.nn.functional.interpolate(image, (512, 512), mode='bilinear')
        real_images = image/255*2-1
        real_images = torch.nn.functional.interpolate(real_images, (512, 512), mode='bilinear')
        image = torch.nn.functional.interpolate(image, (256, 256), mode='bilinear')
        with torch.no_grad():
            target_features = self.vgg16(image, resize_images=False, return_lpips=True) #?

        
        _, c = self.face_recon_model((real_images+1)/2)

        # c = c_3[0]
        # c2w = angle2cam(code_dict['pose']).reshape(1, 16)
        # c = torch.cat([c2w, torch.from_numpy(DEFAULT_INTRINSICS).cuda().reshape(1, 9)], dim=1)
        # c[0,[3,7,11]] = c[0,[3,7,11]]-torch.tensor([0,0.006,0.161]).to(c)
        # print((((c[0,[3,7,11]]-torch.tensor([0,0.006,0.161]).to(c))**2).sum())**(0.5)-2.7)
        # import pdb
        # pdb.set_trace()
        # print(c)
        if cs!=None:
            c = cs
        # c = encode_camera_params(
        #     Pose(matrix_or_rotation=code_dict['pose'][:, :3].cpu().numpy()[0], translation=(0, 0, 3.5), pose_type=PoseType.CAM_2_WORLD,
        #          camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)
        # c = torch.from_numpy(c).cuda().unsqueeze(0)
        # code_dict['pose'][:, :3] *= 0

        ws = self.w_avg
        if w_start!=None:
            start_w = w_start
        else:
            start_w = ws.clone().repeat(1, self.G.backbone.mapping.num_ws, 1)
        w_opt = torch.nn.Parameter(start_w,requires_grad=True)
        
        # Setup noise inputs.
        noise_bufs = {name: buf for (name, buf) in self.G.named_buffers() if 'noise_const' in name and 'synthesis' in name}
        for buf in noise_bufs.values():
            buf = buf.detach()
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        # z = self.z.clone() 
        # z_opt = torch.nn.Parameter(z)

        optimizer = torch.optim.Adam([w_opt] , betas=(0.9, 0.999),#+ list(noise_bufs.values()
                                    lr=self.initial_learning_rate,)
        # optimizer = torch.optim.Adam([z_opt] , betas=(0.9, 0.999),
        #                             lr=self.initial_learning_rate,)            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_iter, eta_min=0.0)               

        # c = c.repeat(self.batch_size, 1)
        # verts = verts.repeat(self.batch_size, 1, 1)
        # exp = code_dict["exp"].repeat(self.batch_size, 1)
        # pose = code_dict["pose"].repeat(self.batch_size, 1)
        
        synthesis_params = {
            'c': c,
        }
        mapping_params = {
            'c': self.c_front,
        }
        result['mapping_params'] = mapping_params
        result['synthesis_params'] = synthesis_params
        # image = image.repeat(self.batch_size, 1, 1, 1)
        images = []
        tbar = tqdm(range(self.num_iter), ncols=80, disable=not verbose, desc="[Inv]Optimizing")
        
        for i in tbar:
            # Noise and learning rate schedule.
            t = i / self.num_iter
            w_noise_scale = self.w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            # z_noise_scale = self.z_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            # z_noise = torch.randn_like(z_opt.repeat(self.batch_size, 1)) * z_noise_scale
            # ws = self.G.mapping(z_opt+z_noise, self.c_front.repeat(1, 1))
            w_noise = torch.randn_like(w_opt.repeat(self.batch_size, 1, 1)) * w_noise_scale
            if w_start!=None:
                ws = w_opt
            else:
                ws = (w_opt + w_noise)
            # print(ws.shape,synthesis_params['c'].shape)
            # print(ws)
            # import pdb
            # pdb.set_trace()
            ret = self.G.synthesis(ws, noise_mode='const', **synthesis_params)
            pred_img = ret['image'].clamp(-1.0+1e-8, 1.0-1e-8)
            
            # Calculate loss.
            loss = 0.
            msg = "[Inv]"
            generated_images=(pred_img.clone())
            pred_img = pred_img.add(1).mul(255/2)
            pred_img = torch.nn.functional.interpolate(pred_img, (256, 256), mode='bilinear')
            pred_features = self.vgg16(pred_img, resize_images=False, return_lpips=True)
            dist = (pred_features - target_features).square().sum()
            loss += dist
            loss_, lpips_loss,_,_ = self.calc_loss(generated_images, real_images, self.G, w_opt,
                                  synthesis_params=synthesis_params,
                                  mapping_params=mapping_params,
                                  use_ball_holder=False)
            loss +=loss_
            msg += f"[PTI]Loss_: {loss_.item():.3f}"
            if lpips_loss is not None:
                msg += f"; LPIPS: {lpips_loss.item():.3f}"
            msg += f"Dist:{dist.item():.3f}; "
            noise_reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]
                while True:
                    noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    noise_reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            msg += f"NoiseReg:{noise_reg_loss.item():.3f}; "
            # loss += noise_reg_loss * self.noise_reg_loss_weight
            msg += f"Total:{loss.item():.3f}; "
            tbar.set_description(msg)
        
            # Update
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if i % self.log_images_step == 0:
                images.append(ret['image'].detach().cpu()[0])
        
        result['w'] = ws.detach()
        result['image'] = ret['image'].detach().cpu()[0]
        result['images'] = images
        result['c'] = c
        # result['z'] = z_opt
        result['c_front'] = self.c_front
        # result['image_target'] = image
        return result
    def calc_loss(self, generated_images, real_images, new_G, w_batch, mapping_params={}, synthesis_params={}, use_ball_holder=False):
        loss = 0.0
        
        if self.l2_lambda > 0:
            l2_loss = F.mse_loss(generated_images, real_images)
            loss += l2_loss * self.l2_lambda
        
        if self.lpips_lambda > 0 or self.lpips_threshold > 0:
            lpips_loss = self.lpips_loss(generated_images, real_images)
            lpips_loss = torch.mean(torch.squeeze(lpips_loss))
            loss += lpips_loss * self.lpips_lambda
        if self.percep_lambda > 0 :
            percep_loss = self.percep_loss(generated_images, real_images.to(generated_images),normalize=True)* self.percep_lambda
            loss += percep_loss
        if use_ball_holder:
            loss += self.space_regulizer.space_regulizer_loss(new_G, w_batch, mapping_params=mapping_params, synthesis_params=synthesis_params)
        if self.sketch_gen:
            sketch_out = self.sketch_gen(generated_images)
            sketch = self.sketch_gen(real_images)
            l2_sketch_loss = F.mse_loss(sketch_out, sketch)
            lpips_sketch_loss = self.lpips_loss(sketch_out, sketch)
            lpips_sketch_loss = torch.mean(torch.squeeze(lpips_sketch_loss))
            loss += 10000*(l2_sketch_loss + lpips_sketch_loss)
        return loss, lpips_loss if self.lpips_threshold > 0 else None,l2_loss,percep_loss,#l2_sketch_loss,lpips_sketch_loss
#===================== Losses

class SpaceRegulizer():
    def __init__(self, original_G, lpips_loss, l2_lambda=0.1, lpips_lambda=0.1, morphing_regulizer_alpha=30):
        self.original_G = original_G
        self.lpips_loss = lpips_loss
        self.morphing_regulizer_alpha = morphing_regulizer_alpha
        self.l2_lambda = l2_lambda
        self.lpips_lambda = lpips_lambda

    def get_morphed_w_code(self, new_w_code, fixed_w):
        return self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code


    def space_regulizer_loss(self, new_G, w_batch, mapping_params={}, synthesis_params={}, num_of_sampled_latents=1):
        loss = 0.0

        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(w_batch.device),
                                            truncation_psi=0.7, **mapping_params)
        territory_indicator_ws = self.get_morphed_w_code(w_samples, w_batch)

        for w_code in territory_indicator_ws:
            w_code = w_code.unsqueeze(0)
            new_img = new_G.synthesis(w_code, noise_mode='const', force_fp32=True, **synthesis_params)['image'].clamp(-1.0, 1.0)
            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, noise_mode='const', force_fp32=True, **synthesis_params)['image'].clamp(-1.0, 1.0)

            if self.l2_lambda > 0:
                l2_loss_val = F.mse_loss(old_img, new_img)
                loss += l2_loss_val * self.l2_lambda

            if self.lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * self.lpips_lambda

        return loss / len(territory_indicator_ws)
    

