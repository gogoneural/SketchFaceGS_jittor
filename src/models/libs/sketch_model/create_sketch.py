import os
import cv2
import numpy as np
import argparse

import hashlib
import platform
import subprocess
# import urllib.request
import jittor as jt
from jittor import nn
# from torch.utils.serialization import load_lua
from src.models.libs.sketch_model.src.model import module
# import torchfile
def sobel(img):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)#3
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)#3
    return cv2.bitwise_or(opImgx, opImgy)

def sketch(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    invImg = 255 - frame
    edgImg0 = sobel(frame)
    edgImg1 = sobel(invImg)
    edgImg = cv2.addWeighted(edgImg0, 0.75, edgImg1, 0.75, 0)
    opImg = 255 - edgImg
    return opImg
def tensor_to_cv2(tensor):
    """ 将归一化到[0,1]的B×C×H×W张量转为OpenCV图像
    Args:
        tensor: jittor Var (C,H,W) 范围[0,1]的float32
    Returns:
        cv_img: (H,W,C) BGR格式的uint8 numpy数组
    """
    if isinstance(tensor, jt.Var):
        img = tensor.permute(1, 2, 0).numpy()
    else:
        img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
def get_sketch_image(image):
    # original = cv2.imread(image_path)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sketch_image = sketch(original)
    return sketch_image[:, :, np.newaxis]

def download_model(modelname, filename, fileurl, filemd5):
    '''
    download the sketch simplification model (sketch_gan.t7) from official implementation: 
    https://github.com/bobbens/sketch_simplification
    '''
    # check if the model already exists
    if os.path.isfile(filename):
        print (filename)
        print(f"Model '{modelname}' already exists. Skipping download.")
    else:
        #  call wget command to download the file 
        print(f"Downloading the sketch simplification {modelname} model...")
        # urllib.request.urlretrieve(fileurl, filename)
        subprocess.call(["wget", "-q", "--show-progress", "--continue", "-O", filename, "--", fileurl])

        # verify the MD5 checksum
        print("checking integrity (md5sum)...")
        with open(filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 != filemd5:
            print("Integrity check failed. File is corrupt!")
            print(f"Try running this script again and if it fails remove '{filename}' before trying again.")
            os.remove(filename)
            raise ValueError("MD5 checksum does not match")
           
        print(f"Model '{modelname}' downloaded successfully")

# if __name__ == "__main__":
class SketchSimplifier:#(nn.Module):
    # define a dictionary to store the model information
    
    # parse command-line arguments
    def __init__(self, device=None):
        # super().__init__()
        # parser = argparse.ArgumentParser(description="sketch data generation")
        # parser.add_argument("modelname", type=str, choices=models.keys(), help="name of the model to download")
        # parser.add_argument("use_cuda", type=bool, default=True, help="use cuda or not")
        # parser.add_argument("data_path", type=str, default="celeba_image", help="path to read images")
        # parser.add_argument("save_path", type=str, default="celeba_sketch", help="path to save sketches")
        # args = parser.parse_args()

        # download the specified model
        models = {
        "GAN": {
            "filename": "model_gan.t7",
            "fileurl": "https://esslab.jp/~ess/data/sketch_gan.t7",
            "filemd5": "3a5b4088f2490ca4b8140a374e80c878"
        },
        "MSE": {
            "filename": "model_mse.t7",
            "fileurl": "https://esslab.jp/~ess/data/sketch_mse.t7",
            "filemd5": "12317df9a0a2a7220629f5f361b45b82"
        },
        "PENCIL(1)": {
            "filename": "model_pencil1.t7",
            "fileurl": "https://esslab.jp/~ess/data/pencil_artist1.t7",
            "filemd5": "33d553ff3a50d6522e79a73002b0025c"
        },
        "PENCIL(2)": {
            "filename": "model_pencil2.t7",
            "fileurl": "https://esslab.jp/~ess/data/pencil_artist2.t7",
            "filemd5": "537b3ad9d46b2a82b65883be747a7ba9"
        }
        }

        print("downloading pretrained models...")
        # model = models['PENCIL(2)']
        # download_model('PENCIL(2)', model["filename"], model["fileurl"], model["filemd5"])
        # print("download finished!")

        # load the model
        # cache = torchfile.load('./face_getsketch_test/pencil_artist2.t7') # sketch_gan.t7
        # self.model = cache[b'model']
        # self.immean = cache[b'mean']
        # self.imstd = cache[b'std']
        
# 或者如果是保存成了 dict
        self.model = module.Net()
        # Load from pre-exported numpy pkl (no torch needed)
        import pickle as _pickle
        import os as _os
        _ckpt_path = _os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..', 'assets', 'model_gan_numpy.pkl')
        _ckpt_path = _os.path.abspath(_ckpt_path)
        with open(_ckpt_path, 'rb') as _f:
            _state_np = _pickle.load(_f)
        self.model.load_state_dict({k: jt.array(v) for k, v in _state_np.items()})
        self.immean  = 0.9664114577640158
        self.imstd  = 0.0858381272736797
        self.model.eval()
    def __call__(self, images, get_sketch=False):

        pred = []
        if get_sketch == True:
            for idx, image in enumerate(images):
                # if idx % 50 == 0:
                #     print("{} out of {}".format(idx, len(images)))
                image = tensor_to_cv2(image)
                data = get_sketch_image(image)
                # Convert numpy HWC to jittor CHW tensor
                data_np = data.astype(np.float32) / 255.0
                data_jt = jt.array(data_np).permute(2, 0, 1)  # HWC→CHW
                data_jt = ((data_jt - self.immean) / self.imstd).unsqueeze(0)
                pred.append(self.model(data_jt).float32())
            return jt.concat(pred, 0)
        data = ((images - self.immean) / self.imstd)
        pred = self.model(data)
        return pred