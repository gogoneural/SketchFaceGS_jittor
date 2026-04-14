'''
Default config for FLAME
'''
from yacs.config import CfgNode as CN
import os

cfg = CN()

cfg = CN()
cfg.base_dir = os.path.dirname(os.path.abspath(__file__))
cfg.topology_path = os.path.join(cfg.base_dir, 'data', 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.dense_template_path = os.path.join(cfg.base_dir, 'data', 'texture_data_256.npy')
cfg.fixed_displacement_path = os.path.join(cfg.base_dir, 'data', 'fixed_displacement_256.npy')
cfg.flame_model_path = os.path.join(cfg.base_dir, 'data', 'generic_model_stable.pkl') 
cfg.flame_lmk_embedding_path = os.path.join(cfg.base_dir, 'data', 'landmark_embedding.npy') 
cfg.face_mask_path = os.path.join(cfg.base_dir, 'data', 'uv_face_mask.png') 
cfg.face_eye_mask_path = os.path.join(cfg.base_dir, 'data', 'uv_face_eye_mask.png') 
cfg.mean_tex_path = os.path.join(cfg.base_dir, 'data', 'mean_texture.jpg') 
cfg.tex_path = os.path.join(cfg.base_dir, 'data', 'FLAME_albedo_from_BFM.npz') 
cfg.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.uv_size = 256
cfg.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.n_shape = 100
cfg.n_tex = 50
cfg.n_exp = 50
cfg.n_cam = 3
cfg.n_pose = 6
cfg.n_light = 27
cfg.use_tex = True
cfg.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# face recognition
cfg.fr_path = os.path.join(cfg.base_dir, 'data', 'resnet50_ft_weight.pkl')

## details
cfg.n_detail = 128
cfg.max_z = 0.01