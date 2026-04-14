import pdb
from typing import Optional

import numpy as np
import torch
from eg3d.torch_utils import misc
from eg3d.torch_utils.ops import upfirdn2d
from eg3d.training.networks_stylegan2 import MappingNetwork, SynthesisLayer, ToRGBLayer, Conv2dLayer
from torch import nn
import torch.nn.functional as F
import  os

class GGHSynthesisBlock(nn.Module):

    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of output color channels.
                 is_last,  # Is this the last block?
                 layer,
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 use_fp16=False,  # Use FP16 for this block?
                 fp16_channels_last=False,  # Use channels-last memory format with FP16?
                 fused_modconv_default=True,  # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
                 **layer_kwargs,  # Arguments for SynthesisLayer.
                 ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.layer=layer 

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                        resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        # self.res_fusion = BasicBlock(out_channels, out_channels)
        # print(f"[GGHBlock] __init__(): res_fusion created, out_ch={out_channels}")
        # pdb.set_trace()
        # self.conv2 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
        #                             conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.torgb_sketch = Two1x1Conv(in_channels=out_channels,out_channels=img_channels,hidden_channels=out_channels)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                                    resample_filter=resample_filter, channels_last=self.channels_last)
        self.w_sketch = nn.Parameter(torch.randn([1,512]), requires_grad=True)
    def forward(self, x, img, img_sketch, ws, w_rgb=None, x_block=None,mask_gt=None,conditions=None,force_fp32=False, fused_modconv=None, update_emas=False, alpha_new_layers: float = 1, **layer_kwargs):
        _ = update_emas  # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            if mask_gt!=None:
                mask_gt_less = mask_gt[:,[1],]
                mask_gt = mask_gt[:,[0],]
                # if self.is_last or self.layer==5:
                x_block1 = x_block[:,:(x_block.shape[1]-22)//2]
                x_block2 = x_block[:,(x_block.shape[1]-22)//2:-22]
                x_block3 = x_block[:,-22:]
                # else:
                    # x_block1 = x_block[:,:x_block.shape[1]//2]
                    # x_block2 = x_block[:,x_block.shape[1]//2:]
                # mask_gt_ = torch.nn.functional.interpolate(
                #     mask_gt.float(), 
                #     size=(x.shape[-2], x.shape[-1]), 
                #     mode='bilinear', 
                #     align_corners=False  # 或 True，看你对角点对齐的需求
                # )  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
                mask_gt_ = (mask_gt > 0.4)   # 或者用 >= 0.4
                # mask_gt_ = mask_gt 
                # x = mask_gt_ * x + ~mask_gt_ * x_block1
                x = mask_gt * x + (1-mask_gt) * x_block1
            x__ = x.clone() 
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

            if mask_gt!=None:
                # mask_gt_ = torch.nn.functional.interpolate(
                #         mask_gt.float(), 
                #         size=(x.shape[-2], x.shape[-1]), 
                #         mode='bilinear', 
                #         align_corners=False  # 或 True，看你对角点对齐的需求
                #     )  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
                mask_gt_ = (mask_gt > 0.35)   # 或者用 >= 0.35
                # mask_gt_ = mask_gt 
                # x = mask_gt_ * x + ~mask_gt_ * x_block2
                x = mask_gt * x + (1-mask_gt) * x_block2
              
            x_=x.clone()
            # x_rgb = x.clone() 
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            # x_ = x.clone() 
            # if mask_gt!=None:
         
            #     # mask_gt_ = torch.nn.functional.interpolate(
            #     #     mask_gt.float(), 
            #     #     size=(x.shape[-2], x.shape[-1]), 
            #     #     mode='bilinear', 
            #     #     align_corners=False  # 或 True，看你对角点对齐的需求
            #     # )  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
            #     # mask_gt_ = (mask_gt_ < 0.7)   # 或者用 >= 0.5
            #     x = mask_gt * x + (1-mask_gt) * x_block
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            # if mask_gt!=None:
            #     mask_gt_less = mask_gt[:,[1],]
            #     mask_gt = mask_gt[:,[0],]
            #     if self.is_last:
            #         x_block1 = x_block[:,:(x_block.shape[1]-22)//2]
            #         x_block2 = x_block[:,(x_block.shape[1]-22)//2:-22]
            #         x_block3 = x_block[:,-22:]
            #     else:
            #         x_block1 = x_block[:,:x_block.shape[1]//2]
            #         x_block2 = x_block[:,x_block.shape[1]//2:]
                # mask_gt_ = torch.nn.functional.interpolate(
                #     mask_gt.float(), 
                #     size=(x.shape[-2], x.shape[-1]), 
                #     mode='bilinear', 
                #     align_corners=False  # 或 True，看你对角点对齐的需求
                # )  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
                # mask_gt_ = (mask_gt > 0.4)   # 或者用 >= 0.4
                # # mask_gt_ = mask_gt 
                # x = mask_gt_ * x + ~mask_gt_ * x_block1
            # x__ = x.clone() 
            if conditions!=None:
                out_same, out_sft = torch.split(x, int(x.size(1) // 2), dim=1)
                out_sft = out_sft * conditions[0] + conditions[1]
                x = torch.cat([out_same, out_sft], dim=1)
                # out_same, out_sft = torch.split(x, int(x.size(1) // 2), dim=1)
                # conditions[1] = out_same * conditions[1] + out_sft
                # x = torch.cat([conditions[0], conditions[1]], dim=1)
            if mask_gt!=None:
                mask_gt_less = mask_gt[:,[1],]
                mask_gt = mask_gt[:,[0],]
                # if self.is_last or self.layer==5:
                x_block1 = x_block[:,:(x_block.shape[1]-22)//2]
                x_block2 = x_block[:,(x_block.shape[1]-22)//2:-22]
                x_block3 = x_block[:,-22:]
                # else:
                    # x_block1 = x_block[:,:x_block.shape[1]//2]
                    # x_block2 = x_block[:,x_block.shape[1]//2:]
                # mask_gt_ = torch.nn.functional.interpolate(
                #     mask_gt.float(), 
                #     size=(x.shape[-2], x.shape[-1]), 
                #     mode='bilinear', 
                #     align_corners=False  # 或 True，看你对角点对齐的需求
                # )  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
                mask_gt_ = (mask_gt > 0.4)   # 或者用 >= 0.4
                # x = mask_gt_ * x + ~mask_gt_ * x_block1
                x = mask_gt * x + (1-mask_gt) * x_block1
            x__ = x.clone() 
            # if w_rgb!=None:
            #     x = self.conv1(x, w_rgb2.squeeze(1), fused_modconv=fused_modconv, **layer_kwargs)
            # else:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
           
            if conditions!=None:
                out_same, out_sft = torch.split(x, int(x.size(1) // 2), dim=1)
                out_sft = out_sft * conditions[2] + conditions[3]
                x = torch.cat([out_same, out_sft], dim=1)
           

            if mask_gt!=None:
         
                # mask_gt_ = torch.nn.functional.interpolate(
                #     mask_gt.float(), 
                #     size=(x.shape[-2], x.shape[-1]), 
                #     mode='bilinear', 
                #     align_corners=False  # 或 True，看你对角点对齐的需求
                # )  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
                mask_gt_ = (mask_gt > 0.35)   # 或者用 >= 0.35
                # mask_gt_ = mask_gt 
                # x = mask_gt_ * x + ~mask_gt_ * x_block2
                x = mask_gt * x + (1-mask_gt) * x_block2
            x_ = x.clone() 
            
            
            # if mask_gt!=None:
         
            #     # mask_gt_ = torch.nn.functional.interpolate(
            #     #     mask_gt.float(), 
            #     #     size=(x.shape[-2], x.shape[-1]), 
            #     #     mode='bilinear', 
            #     #     align_corners=False  # 或 True，看你对角点对齐的需求
            #     # )  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
            #     mask_gt_ = (mask_gt > 0.35)   # 或者用 >= 0.35
                
            #     x = mask_gt_ * x + ~mask_gt_ * x_block2
               
            # x_ = x.clone() 
        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if img_sketch is not None:
            misc.assert_shape(img_sketch, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img_sketch = upfirdn2d.upsample2d(img_sketch, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
          
            if w_rgb!=None:
                # if  self.is_last or self.layer==5:
                w_rgb2 = next(w_iter)
                y_sketch = self.torgb_sketch(x,)

                y_new = self.torgb(x, w_rgb2.squeeze(1), fused_modconv=fused_modconv)
                y_before = self.torgb(x, w_rgb.squeeze(1), fused_modconv=fused_modconv)
                y_before[:,:-12] = y_new[:,:-12]
                y = y_before
                # w_rgb = next(w_iter)
                # else:
                #     y = self.torgb(x, w_rgb.squeeze(1), fused_modconv=fused_modconv)
                #     y_sketch = self.torgb_sketch(x,)
            else:
                w_rgb = next(w_iter)
               
                y = self.torgb(x, w_rgb, fused_modconv=fused_modconv)
                y_sketch = self.torgb_sketch(x, )
               
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            y_sketch = y_sketch.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            if alpha_new_layers is not None:
                y = alpha_new_layers * y  # Potentially lower contribution of output map if it comes from a newly introduced layer after progressive growing
            img = img.add_(y) if img is not None else y
            img_sketch = img_sketch.add_(y_sketch) if img_sketch is not None else y_sketch
            
            # if  self.is_last and mask_gt!=None:
            #     mask_gt_ = (mask_gt_less > 0.35)   # 或 
            #     img = mask_gt_ * img + ~mask_gt_ * x_block3
                
            # if  self.is_last or self.layer==5:
            x___ = img.clone()
            x_ = torch.cat([x_,x___],1)
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        assert img_sketch is None or img_sketch.dtype == torch.float32
        
        return x, img, x_, x__, img_sketch
    
    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

class GGHSynthesisBlockNoUp(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution,
                conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, alpha_new_layers: float = 1, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        # if img is not None:
            # misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            # img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            if alpha_new_layers is not None:
                y = alpha_new_layers * y  # Potentially lower contribution of output map if it comes from a newly introduced layer after progressive growing

            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

class GGHSynthesisNetwork(nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 pretrained_plane_resolution: Optional[int] = None,  # For progressive Growing
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.pretrained_plane_resolution = pretrained_plane_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_resolution_log2_pretrained = int(np.log2(pretrained_plane_resolution)) if pretrained_plane_resolution is not None else self.img_resolution_log2
        if num_fp16_res > 0:
            # If new layers are added and the previous last n layers had fp16, those should still have fp16 in addition to the new layers that come after
            num_fp16_res += (self.img_resolution_log2 - self.img_resolution_log2_pretrained)
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for layer, res in enumerate(self.block_resolutions):
            is_new_layer = pretrained_plane_resolution is not None and res > pretrained_plane_resolution
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution) or (pretrained_plane_resolution is not None and res == pretrained_plane_resolution)
            block = GGHSynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                      img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, layer=layer, **block_kwargs)
            if is_new_layer:
                # Initialize new layers with 0 torgb, to not disturb the lower resolution output in the beginning
                block.torgb.weight.data.zero_()

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, alpha_new_layers: float = 1,conditions=None,return_xblock=False, **block_kwargs):
        block_ws = []
        # from torch.profiler import record_function
        # with record_function("model_forward0.1"):

        if ws.shape[1]!=14:
            ws2 = ws[:,14:]
            if isinstance(conditions, dict):
                block_ws_rgb = []
            
        # if style_x !=None:
            # ws = torch.zeros_like(ws).to(ws)
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, self.num_ws+self.num_ws/2, self.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0         
                w_idx2 = 0
                for res in self.block_resolutions:
                    block = getattr(self, f'b{res}')
                    block_ws.append(torch.cat([ws.narrow(1, w_idx, block.num_conv ),ws2.narrow(1, w_idx2, block.num_torgb)],1))
                    if isinstance(conditions, dict):
                        if conditions['w'].shape[1]==14:          
                            block_ws_rgb.append(conditions['w'].narrow(1, w_idx+block.num_conv, block.num_torgb))
                        else:
                            block_ws_rgb.append(conditions['w'].narrow(1, 14+w_idx2, block.num_torgb))
                    w_idx += block.num_conv
                    w_idx2 += block.num_torgb

        else:
            if isinstance(conditions, dict):
                block_ws_rgb = []
            
            # if style_x !=None:
                # ws = torch.zeros_like(ws).to(ws)
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0      
                w_idx2 = 0   
                for res in self.block_resolutions:
                    block = getattr(self, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    if isinstance(conditions, dict):
                        if conditions['w'].shape[1]==14:          
                            block_ws_rgb.append(conditions['w'].narrow(1, w_idx+block.num_conv, block.num_torgb))
                        else:
                            block_ws_rgb.append(conditions['w'].narrow(1, 14+w_idx2, block.num_torgb))
                        
                    w_idx += block.num_conv
                    w_idx2 += block.num_torgb
    # with record_function("model_forward0.2"):   
        x_block = []
    
        x = img = img_sketch = None
        if isinstance(conditions, dict):
            self.block_resolutions.reverse()
            mask_gt = []
            # mask_gt.append(conditions['mask_gt'])
            for i,res in enumerate(self.block_resolutions):
                if i!=0 :
                    mask_gt.append( torch.nn.functional.interpolate(
                            mask_gt[i-1].float(), 
                            size=(res, res), 
                            #  mode='bilinear',
                            # align_corners=False 
                             mode='area'  # 或 True，看你对角点对齐的需求
                        ))  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
                    
                    # mask_gt.append(torch.cat([F.max_pool2d(conditions['mask_gt_last'].float(), kernel_size=512//res, stride=512//res),
                    #                           1-F.max_pool2d(1-conditions['mask_gt_last'].float(), kernel_size=512//res, stride=512//res)],1))
                else:
                    mask_gt.append( torch.cat([torch.nn.functional.interpolate(
                            conditions['mask_gt_last'].float(), 
                            size=(res, res), 
                            #  mode='bilinear',
                            # align_corners=False 
                             mode='area' # 或 True，看你对角点对齐的需求
                        ),torch.nn.functional.interpolate(
                            conditions['mask_gt'].float(), 
                            size=(res, res), 
                        #    mode='bilinear',
                            # align_corners=False 
                             mode='area' # 或 True，看你对角点对齐的需求
                        )],1))  # 结果是 [B,1,h,
                    # mask_gt.append(torch.cat([F.max_pool2d(conditions['mask_gt_last'].float(), kernel_size=512//res, stride=512//res),
                    #                           1-F.max_pool2d(1-conditions['mask_gt_last'].float(), kernel_size=512//res, stride=512//res)],1))

                # if i!=0:
                #     mask_gt.append( torch.nn.functional.interpolate(
                #             mask_gt[i-1].float(), 
                #             size=(res, res), 
                #             mode='bilinear', 
                #             align_corners=False  # 或 True，看你对角点对齐的需求
                #         ))  # 结果是 [B,1,h,w]，值会在 [0,1] 之间
                # else:
                #     mask_gt.append( torch.nn.functional.interpolate(
                #             conditions['mask_gt_last'].float(), 
                #             size=(res, res), 
                #             mode='bilinear', 
                #             align_corners=False  # 或 True，看你对角点对齐的需求
                #         ))  # 结果是 [B,1,h,
            self.block_resolutions.reverse()
            mask_gt.reverse()
    # with record_function("model_forward0.3"):   
        for idx,(res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            block = getattr(self, f'b{res}')
            if self.pretrained_plane_resolution is not None and res > self.pretrained_plane_resolution:
                x, img = block(x, img, cur_ws, alpha_new_layers=alpha_new_layers, **block_kwargs)
            else:
                if conditions!=None and idx!=0:
                    if isinstance(conditions, dict):
                        if idx==6:
                            x, img, x_, x__,img_sketch = block(x, img, img_sketch, cur_ws, conditions=conditions['conditions'][4*idx-4:4*idx],mask_gt=mask_gt[idx], 
                                    x_block=conditions['x_block'][idx],w_rgb = block_ws_rgb[idx],**block_kwargs)
                        else:
                            x, img, x_, x__,img_sketch = block(x, img, img_sketch, cur_ws, conditions=conditions['conditions'][4*idx-4:4*idx],mask_gt=mask_gt[idx], 
                                    x_block=conditions['x_block'][idx],w_rgb = block_ws_rgb[idx],**block_kwargs)
                    else:
                        x, img, x_, x__,img_sketch = block(x, img, img_sketch, cur_ws, conditions=conditions[4*idx-4:4*idx], **block_kwargs)
                else:
                    if isinstance(conditions, dict):
                        x, img, x_, x__,img_sketch = block(x, img, img_sketch, cur_ws, mask_gt=mask_gt[idx], x_block=conditions['x_block'][idx],w_rgb = block_ws_rgb[idx],**block_kwargs)
                    else:    
                        x, img, x_, x__,img_sketch = block(x, img, img_sketch, cur_ws,  **block_kwargs)
            x_block.append(torch.cat([x__,x_],1))
            
        if return_xblock:
            return {'x_block':x_block}, img, img_sketch
        return img, img_sketch

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


class GGHGenerator(nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 pretrained_plane_resolution: Optional[int] = None,  # For progressive Growing
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = GGHSynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                             pretrained_plane_resolution=pretrained_plane_resolution,
                                             **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # print(ws.shape)
        # pdb.set_trace()
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img


class Two1x1Conv(nn.Module):
    """
    两层 1×1 卷积构成的模块，中间使用 LeakyReLU 激活。
    Args:
        in_channels (int):  输入通道数
        hidden_channels (int): 第一层卷积的输出（隐藏）通道数
        out_channels (int): 输出通道数
        negative_slope (float): LeakyReLU 的负半轴斜率，默认 0.2
    """
    def __init__(self, in_channels, hidden_channels, out_channels, negative_slope=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  hidden_channels, kernel_size=1, bias=True)
        self.act   = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels,   kernel_size=1, bias=True)

        # 权重初始化（可选）
        nn.init.kaiming_uniform_(self.conv1.weight, a=negative_slope)
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_uniform_(self.conv2.weight, a=negative_slope)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        """
        x: Tensor of shape (N, in_channels, H, W)
        returns: Tensor of shape (N, out_channels, H, W)
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x