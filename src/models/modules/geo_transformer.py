# Copyright (C) 2024-present Alibaba yuanjing aigclib Corporation. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
import pdb
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from linformer import Linformer
# from .curope3d import cuRoPE3D
from core.models.modules.deformable_cross_attention import MSDCAWrapper
import random

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    dropout: float = 0.1
    grid_size: int = 128
    n_points: int = 5023
    max_batch_size: int = 32

    deform_att_cfg: dict = field(default_factory=dict)
    deform_att_cfg2: dict = field(default_factory=dict)
class RoPE1D(torch.nn.Module):

    def __init__(self, freq=10000.0, F0=1.0, scaling_factor=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.scaling_factor = scaling_factor
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens (t position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        tokens = tokens
        D = tokens.size(3)
        assert positions.ndim == 2  # Batch, Seq
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        tokens = self.apply_rope1d(tokens, positions, cos, sin)
        return tokens


class SinusoidalPositionEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionEncoding, self).__init__()
        self.d_model = d_model

        # 生成位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # (d_model / 2)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)  # sin
        pe[:, 1::2] = torch.cos(position * div_term)  # cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input:
            x: (batch_size, seq_len, d_model)
        Output:
            x with added position encoding (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        self.rope1d = RoPE1D()
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_p = args.dropout

        self.xyzs_idx = torch.arange(args.n_points+64**2+16**2+1)
        # self.xyzs_idx = torch.arange(args.n_points )
    def forward(
        self,

        h: torch.Tensor,
    ):
        xyzs_idx = self.xyzs_idx.reshape(1,-1).repeat(h.shape[0],1).to(h.device)
        bsz, seqlen, _ = h.shape
        xq, xk, xv = self.wq(h), self.wk(h), self.wv(h)

        xq = xq.view(bsz,  seqlen, self.n_heads, self.head_dim).permute(0,2,1,3)
        xk = xk.view(bsz, seqlen,  self.n_heads,   self.head_dim).permute(0,2,1,3)
        xv = xv.view(bsz,  seqlen, self.n_heads,  self.head_dim).permute(0,2,1,3)
        xq = self.rope1d(xq, xyzs_idx)
        xk = self.rope1d(xk, xyzs_idx)

        # if xq.dtype == torch.float16 or xq.dtype == torch.bfloat16:
        qkv = torch.stack([xq, xk, xv], dim=2).to(torch.float16)#bxhx3xnxd
        output = flash_attn_qkvpacked_func(qkv=qkv, dropout_p=self.dropout_p if self.training else 0)
        output = output.view(bsz, seqlen, -1).to(torch.float32)
        

        # scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores = self.dropout(scores)
        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeoTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, n_layers: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attention = SelfAttention(args)
        # self.self_attention =  Linformer(
        #     dim=self.dim,
        #     seq_len=9376,
        #     depth=1,
        #     heads=self.n_heads,
        #     k=256
        # )

        self.pos_embedding = SinusoidalPositionEncoding(self.dim,9376)
        self.cross_attention = MSDCAWrapper(

            dropout=args.dropout,
            **args.deform_att_cfg
        )
        self.cross_attention2 = MSDCAWrapper(
            dropout=args.dropout,
            **args.deform_att_cfg
        )
        self.n_points = args.n_points
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.self_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        self.embedding_layer = nn.Embedding(3, args.dim)
        self.n_layers = n_layers
    def forward(
        self,

        latents: torch.Tensor,
        images: torch.Tensor,
        visible_verts: torch.Tensor,
        feature_global:torch.Tensor,
        img_back:torch.Tensor,
        level_start_index2:torch.Tensor,
        spatial_shapes2:torch.Tensor,
        reference_points_cam:torch.Tensor,
        reference_points_cam_sym: torch.Tensor,
        remaining_sym_verts: torch.Tensor,
        remaining_nosym_verts: torch.Tensor,
        **kwargs,
    ):
        # all_verts = torch.arange(self.n_points).to(latents)
        # remaining_verts = all_verts[~torch.isin(all_verts, visible_verts)].long()
        # remaining_sym_verts = remaining_verts[torch.isin(remaining_verts, visible_verts_sym)]

        # label = torch.zeros([latents.shape[0],latents.shape[1]])
        # label[:,visible_verts] = 1
        # label[:, remaining_sym_verts] = 2
        # label_embed = self.embedding_layer(label.long().to(latents.device))

        # h = self.self_attention_norm(latents)#feature_global
        # h = h + self.dropout(self.self_attention( self.pos_embedding(torch.cat([h,feature_global],-2))))[:,:h.shape[1],:]

        h = self.self_attention_norm(latents)  # feature_global
        h = h + self.dropout(self.self_attention(self.pos_embedding(torch.cat([h, feature_global], -2))))[:,
                :h.shape[1], :]

        h = self.cross_attention_norm(h.float())

        # h = self.pos_embedding(h)

        a = self.cross_attention(
            query=h,
            value=images.float(),
            visible_verts=visible_verts,
            reference_points_cam=reference_points_cam,
            **kwargs
        ).to(latents)
        # if self.layer_id>=self.n_layers//2:
        # b = self.cross_attention2(
        #     query=query,
        #     value=img_back.float(),
        #     visible_verts=remaining_verts,
        #     reference_points_cam =reference_points_cam,
        #     level_start_index=level_start_index2,
        #     spatial_shapes=spatial_shapes2,
        # ).to(latents)
        # else:
        # remaining_verts = remaining_verts[torch.isin(remaining_verts, visible_verts_sym)]

        b = self.cross_attention2(
            query=h,
            value=images.float(),
            visible_verts=remaining_sym_verts,
            reference_points_cam=reference_points_cam_sym,
            **kwargs
        ).to(latents)
        # c = self.cross_attention(
        #             query=h,
        #             value=img_back.float(),
        #             visible_verts=remaining_nosym_verts,
        #             reference_points_cam=reference_points_cam,
        #             **kwargs
        #             ).to(latents)
        # updated_h = h.clone()
        #         # updated_h[:, visible_verts, :] = a
        # a = a + self.dropout(self.feed_forward(self.ffn_norm(a)))
        # #
        # b = b + self.dropout(self.feed_forward(self.ffn_norm(b)))

        h = h.scatter(1, visible_verts[None, :, None].expand(-1, -1, h.size(-1)), a)
        h = h.scatter(1, remaining_sym_verts[None, :, None].expand(-1, -1, h.size(-1)), b)
        # h = h.scatter(1, remaining_nosym_verts[None, :, None].expand(-1, -1, h.size(-1)), c)

        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))

        return out

class SelfTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, n_layers: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attention = SelfAttention(args)

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.self_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        self.embedding_layer = nn.Embedding(3, args.dim)
        self.n_layers = n_layers
    def forward(
        self,

        latents: torch.Tensor,

        feature_global:torch.Tensor,


        **kwargs,
    ):


        h = self.self_attention_norm(latents)#feature_global
        h = h + self.dropout(self.self_attention( torch.cat([h,feature_global],-2)))[:,:h.shape[1],:]
        # h = h + self.dropout(self.self_attention(h))

        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))

        return out

class SelfTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.grid_size = params.grid_size

        self.query_token = nn.Parameter(torch.randn(1, 1, params.dim - 3))

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(SelfTransformerBlock(layer_id, self.n_layers,params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # self.pos_embedding = SinusoidalPositionEncoding(params.dim, 5023)

        self.embedding_layer = nn.Embedding(2, params.dim)

        self.view_mlp = CustomMLP(9, [64, 64], 64)

        self.view_mlp2 = CustomMLP(params.dim + 64, [params.dim, params.dim], params.dim)

        self.mask_token = nn.Parameter(torch.randn(1,1, params.dim ))
    def forward(self, h, visible_verts, feature_global, view1,**kwargs):
        """Constructs 3D latents from input images and camera parameters.

        Args:
            xyzs: [B, S, 3], 3D coordinates of the input points
            img_feats: [B, V, L, C], input hierarchical image features.
                V: number of views
                L: h_0 * w_0 + h_1 * w_1 + ... + h_{num_levels-1} * w_{num_levels-1}
                C: feature dimension
                Refer to Deformable Cross-Attention for more details.
            spatial_shapes: [V, 2], spatial shapes of the hierarchical image features
            proj_matrix: [B, V, 3, 4], world-to-image projection matrices

        Returns:
            latents: [B, S, D], 3D latents of the input points
        """
        view = self.view_mlp(torch.cat([view1],-1))

        view = view[:,None,:].repeat(1,h.shape[1],1)

        h = self.view_mlp2(torch.cat([h,view],-1))

        # all_verts = torch.arange(5023).to(h)
        # remaining_verts = all_verts[~torch.isin(all_verts, visible_verts)].long()
        # remaining_sym_verts = remaining_verts[torch.isin(remaining_verts, visible_verts_sym)]
        # remaining_nosym_verts = remaining_verts[~torch.isin(remaining_verts, visible_verts_sym)]
        label = torch.zeros(h.shape[0],h.shape[1])
        label[:,visible_verts] = 1
        # # label[:, visible_verts2] = 2
        label_embed = self.embedding_layer(label.long().to(h.device))

        h = h + label_embed


        for idx,layer in enumerate(self.layers):
            h = layer(

                latents=h,#Bxnxd

                feature_global=feature_global,



            )


        return h



class GeoTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.grid_size = params.grid_size

        self.query_token = nn.Parameter(torch.randn(1, 1, params.dim - 3))

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(GeoTransformerBlock(layer_id, self.n_layers,params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.pos_embedding = SinusoidalPositionEncoding(params.dim, 5023)

        self.embedding_layer = nn.Embedding(3, params.dim)

        self.self_attention =  Linformer(
            dim = params.dim,
            seq_len =5023 ,
            depth = 8,
            heads = 8,
            k = 256,
            one_kv_head = False,
            share_kv = False
        )

        self.mask_token = nn.Parameter(torch.randn(1,1, params.dim ))
    def forward(self, h, uv, img_feats, level_start_index, spatial_shapes,visible_verts, feature_global,feature_img_back,uv_sym,visible_verts_sym,xyz_scale=1.0, **kwargs):
        """Constructs 3D latents from input images and camera parameters.

        Args:
            xyzs: [B, S, 3], 3D coordinates of the input points
            img_feats: [B, V, L, C], input hierarchical image features.
                V: number of views
                L: h_0 * w_0 + h_1 * w_1 + ... + h_{num_levels-1} * w_{num_levels-1}
                C: feature dimension
                Refer to Deformable Cross-Attention for more details.
            spatial_shapes: [V, 2], spatial shapes of the hierarchical image features
            proj_matrix: [B, V, 3, 4], world-to-image projection matrices

        Returns:
            latents: [B, S, D], 3D latents of the input points
        """

        all_verts = torch.arange(5023).to(h)
        remaining_verts = all_verts[~torch.isin(all_verts, visible_verts)].long()
        remaining_sym_verts = remaining_verts[torch.isin(remaining_verts, visible_verts_sym)]
        remaining_nosym_verts = remaining_verts[~torch.isin(remaining_verts, visible_verts_sym)]
        label = torch.zeros(h.shape[0],h.shape[1])
        label[:,visible_verts] = 1
        label[:, remaining_sym_verts] = 2
        label_embed = self.embedding_layer(label.long().to(h.device))

        h = h + label_embed


        reference_points_cam = (uv+1)/2
        reference_points_cam[...,1] = reference_points_cam[...,1]
        reference_points_cam_sym = (uv_sym + 1) / 2
        # latents = h.clone()
        # h = h[:,visible_verts]
        # reference_points_cam = reference_points_cam[:,visible_verts]
        # level_start_index = torch.tensor([0]).to(reference_points_cam.device)
        # spatial_shapes = torch.tensor([[512,512]]).to(reference_points_cam.device)
        for idx,layer in enumerate(self.layers):
            h = layer(

                latents=h,#Bxnxd
                images=img_feats,#Bx(HW)xd
                reference_points_cam=reference_points_cam,#Bxnx2
                spatial_shapes=spatial_shapes,#lx2
                level_start_index=level_start_index,
                visible_verts=visible_verts,# l
                feature_global=feature_global,
                img_back=feature_img_back,
                level_start_index2=torch.tensor([0, 512 * 512,512*512+64*64]).to(h.device),
                spatial_shapes2=torch.tensor([[512, 512],[64,64],[16,16]]).to(h.device),
                reference_points_cam_sym=reference_points_cam_sym,
                remaining_sym_verts=remaining_sym_verts,
                remaining_nosym_verts=remaining_nosym_verts,

            )
        # latents[:,visible_verts] = h.clone()
        # batch_size,num_verts=h.shape[:2]
        # h = self.norm(h)


        # mask_ratio = 0#random.uniform(0., 0.7)
        # num_masked = int(num_verts * mask_ratio)  # 每个batch中有多少个元素被mask
        # # 生成一个随机掩码，随机选择30%的节点位置
        # mask = torch.rand(batch_size, num_verts) < mask_ratio # 生成一个布尔掩码
        # mask = mask.to(h.device)  # 确保掩码在同一设备上
        #
        # h[mask] = self.mask_token[0,0]
        # h[:,remaining_nosym_verts] = self.mask_token
        # h = self.pos_embedding(h)

        # h = h + label_embed

        # h = self.self_attention(h)

        return h


class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomMLP, self).__init__()

        # 创建一个空的模块列表
        layers = []
        in_features = input_size

        # 构建每一层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))  # 创建全连接层
            layers.append(nn.SiLU())  # 添加ReLU激活函数
            in_features = hidden_size  # 更新输入特征数量为当前层的输出尺寸

        # 最后一层输出
        layers.append(nn.Linear(in_features, output_size))  # 输出层，无激活函数

        # 将所有层堆叠成一个模块
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)