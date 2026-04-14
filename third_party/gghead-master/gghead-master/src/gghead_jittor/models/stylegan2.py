import pdb
from typing import Optional

import numpy as np
import jittor as jt
from gghead_jittor.StyleGAN2.training.utils.misc import assert_shape, suppress_tracer_warnings
from gghead_jittor.StyleGAN2.training.utils.ops import upfirdn2d
from gghead_jittor.StyleGAN2.training.networks import MappingNetwork, SynthesisLayer, ToRGBLayer, Conv2dLayer
from jittor import nn
import os

# Compatibility wrapper for misc.assert_shape used throughout
class _MiscCompat:
    assert_shape = staticmethod(assert_shape)
    suppress_tracer_warnings = staticmethod(suppress_tracer_warnings)
misc = _MiscCompat()


def _gaussian_blur_jt(x: jt.Var, ks: int, sigma: float) -> jt.Var:
    """Apply 2D separable gaussian blur, depthwise, to a [B,C,H,W] tensor."""
    coords = jt.array([(i - ks // 2) for i in range(ks)], dtype='float32')
    g = jt.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)  # [ks, ks]
    C = x.shape[1]
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).broadcast([C, 1, ks, ks])  # [C,1,ks,ks]
    pad = ks // 2
    return jt.nn.conv2d(x, kernel, padding=pad, groups=C)


def soft_blend_features(x_src: jt.Var, x_tgt: jt.Var, mask: jt.Var) -> jt.Var:
    """
    Soft alpha blend: gaussian-blur mask for smooth boundary, then lerp.
    x_src: edited-path features (mask=1 region)
    x_tgt: GT-path features   (mask=0 region)
    """
    H = mask.shape[-1]
    if H <= 32:
        ks, sigma = 3, 1.0
    elif H <= 128:
        ks, sigma = 5, 1.5
    else:
        ks, sigma = 7, 2.0
    mask_blurred = _gaussian_blur_jt(mask.float32(), ks, sigma)
    mask_soft = jt.maximum(mask.float32(), mask_blurred)
    return mask_soft * x_src + (1.0 - mask_soft) * x_tgt

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
        self.resample_filter = upfirdn2d.setup_filter(resample_filter)
        self.num_conv = 0
        self.num_torgb = 0
        self.layer=layer 

        if in_channels == 0:
            self.const = jt.nn.Parameter(jt.randn([out_channels, resolution, resolution]))

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
        self.w_sketch = jt.nn.Parameter(jt.randn([1, 512]))
    def execute(self, x, img, img_sketch, ws, w_rgb=None, x_block=None,mask_gt=None,conditions=None,force_fp32=False, fused_modconv=None, update_emas=False, alpha_new_layers: float = 1, **layer_kwargs):
        _ = update_emas  # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = np.float16 if self.use_fp16 and not force_fp32 else np.float32
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.is_training())

        # Input.
        if self.in_channels == 0:
            x = self.const.float32() if dtype == np.float32 else self.const.float16()
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.float32() if dtype == np.float32 else x.float16()

        # Main layers.
        if self.in_channels == 0:
            if mask_gt is not None:
                x_block1 = x_block[:,:(x_block.shape[1]-22)//2]
                x_block2 = x_block[:,(x_block.shape[1]-22)//2:-22]
                x_block3 = x_block[:,-22:]
            x__ = x.clone() 
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

            if mask_gt is not None:
                x = soft_blend_features(x, x_block2, mask_gt)
              
            x_=x.clone()
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y + x
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
            if conditions is not None:
                half_c = x.shape[1] // 2
                out_same, out_sft = x[:, :half_c], x[:, half_c:]
                out_sft = out_sft * conditions[0] + conditions[1]
                x = jt.concat([out_same, out_sft], dim=1)
                # out_same, out_sft = torch.split(x, int(x.size(1) // 2), dim=1)
                # conditions[1] = out_same * conditions[1] + out_sft
                # x = torch.cat([conditions[0], conditions[1]], dim=1)
            if mask_gt is not None:
                x_block1 = x_block[:,:(x_block.shape[1]-22)//2]
                x_block2 = x_block[:,(x_block.shape[1]-22)//2:-22]
                x_block3 = x_block[:,-22:]
            x__ = x.clone() 
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

            if mask_gt is not None:
                x = soft_blend_features(x, x_block2, mask_gt)
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
          
            if w_rgb is not None:
                # if  self.is_last or self.layer==5:
               
                y_sketch = self.torgb_sketch(x)
                    

               
                y = self.torgb(x, w_rgb.squeeze(1), fused_modconv=fused_modconv)
             
                # else:
                #     y = self.torgb(x, w_rgb.squeeze(1), fused_modconv=fused_modconv)
                #     y_sketch = self.torgb_sketch(x,)
            else:
                w_rgb = next(w_iter)
               
                y = self.torgb(x, w_rgb, fused_modconv=fused_modconv)
                y_sketch = self.torgb_sketch(x, )
               
            y = y.float32()
            y_sketch = y_sketch.float32()
            if alpha_new_layers is not None:
                y = alpha_new_layers * y  # Potentially lower contribution of output map if it comes from a newly introduced layer after progressive growing
            img = img + y if img is not None else y
            img_sketch = img_sketch + y_sketch if img_sketch is not None else y_sketch
            
            # if  self.is_last and mask_gt!=None:
            #     mask_gt_ = (mask_gt_less > 0.35)   # 或 
            #     img = mask_gt_ * img + ~mask_gt_ * x_block3
                
            # if  self.is_last or self.layer==5:
            x___ = img.clone()
            x_ = jt.concat([x_,x___],1)
        assert x.dtype == dtype
        assert img is None or img.dtype == np.float32
        assert img_sketch is None or img_sketch.dtype == np.float32
        
        return x, img, x_, x__, img_sketch
    
    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

class GGHSynthesisBlockNoUp(nn.Module):
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
        self.resample_filter = upfirdn2d.setup_filter(resample_filter)
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = jt.nn.Parameter(jt.randn([out_channels, resolution, resolution]))

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

    def execute(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, alpha_new_layers: float = 1, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = np.float16 if self.use_fp16 and not force_fp32 else np.float32
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.is_training())

        # Input.
        if self.in_channels == 0:
            x = self.const.float32() if dtype == np.float32 else self.const.float16()
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.float32() if dtype == np.float32 else x.float16()

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y + x
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.float32()
            if alpha_new_layers is not None:
                y = alpha_new_layers * y  # Potentially lower contribution of output map if it comes from a newly introduced layer after progressive growing

            img = img + y if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == np.float32
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
                block.torgb.weight.assign(jt.zeros_like(block.torgb.weight))

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def execute(self, ws, alpha_new_layers: float = 1,conditions=None,return_xblock=False, **block_kwargs):
        block_ws = []

        if ws.shape[1]!=14:
            ws2 = ws[:,14:]
            if isinstance(conditions, dict):
                block_ws_rgb = []
            
            misc.assert_shape(ws, [None, self.num_ws+self.num_ws/2, self.w_dim])
            ws = ws.float32()
            w_idx = 0         
            w_idx2 = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(jt.concat([ws[:, w_idx:w_idx+block.num_conv], ws2[:, w_idx2:w_idx2+block.num_torgb]], 1))
                if isinstance(conditions, dict):
                    if conditions['w'].shape[1]==14:          
                        block_ws_rgb.append(conditions['w'][:, w_idx+block.num_conv:w_idx+block.num_conv+block.num_torgb])
                    else:
                        block_ws_rgb.append(conditions['w'][:, 14+w_idx2:14+w_idx2+block.num_torgb])
                w_idx += block.num_conv
                w_idx2 += block.num_torgb

        else:
            if isinstance(conditions, dict):
                block_ws_rgb = []
            
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.float32()
            w_idx = 0      
            w_idx2 = 0   
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws[:, w_idx:w_idx+block.num_conv+block.num_torgb])
                if isinstance(conditions, dict):
                    if conditions['w'].shape[1]==14:          
                        block_ws_rgb.append(conditions['w'][:, w_idx+block.num_conv:w_idx+block.num_conv+block.num_torgb])
                    else:
                        block_ws_rgb.append(conditions['w'][:, 14+w_idx2:14+w_idx2+block.num_torgb])
                    
                w_idx += block.num_conv
                w_idx2 += block.num_torgb
    # with record_function("model_forward0.2"):   
        x_block = []
    
        x = img = img_sketch = None
        if isinstance(conditions, dict):
            mask_orig = conditions['mask_gt'].float32()
            mask_gt = []
            for res in self.block_resolutions:
                mask_gt.append(nn.interpolate(
                    mask_orig, size=(res, res), mode='bilinear', align_corners=False
                ))
    # with record_function("model_forward0.3"):   
        for idx,(res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            block = getattr(self, f'b{res}')
            if self.pretrained_plane_resolution is not None and res > self.pretrained_plane_resolution:
                x, img = block(x, img, cur_ws, alpha_new_layers=alpha_new_layers, **block_kwargs)
            else:
                if conditions is not None and idx != 0:
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
            x_block.append(jt.concat([x__,x_],1))
            
        if return_xblock:
            return x_block, img, img_sketch
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

    def execute(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
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
        self.act   = nn.LeakyReLU(scale=negative_slope)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels,   kernel_size=1, bias=True)

        # 权重初始化（可选） - Jittor Conv2d uses kaiming_uniform by default
        # Biases are already initialized to zero by default in Jittor

    def execute(self, x):
        """
        x: Tensor of shape (N, in_channels, H, W)
        returns: Tensor of shape (N, out_channels, H, W)
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x