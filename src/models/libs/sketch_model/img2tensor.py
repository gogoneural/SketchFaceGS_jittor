import time
import torch
import torch.nn as nn
import functools
import os
import datetime

from torchvision import transforms, utils
from PIL import Image, ImageFilter
from torch.autograd import Variable
import networks_tensor

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = Variable(img, requires_grad = True)
    img = img.cuda()
    return img

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    file_names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                file_names.append(fname)

    return images, file_names

transform = transforms.Compose(
        [
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

if __name__ == '__main__':
    gpu_ids = ['0']
    Tensor2Sketch = networks_tensor.define_G(3, 3, 64, 'tensor2sketch', 
                                  4, 4, 1, 
                                  3, 'instance', gpu_ids)
    sketch_gen_checkpoint = '../code/face_getsketch_part/checkpoints/sketch2sketch_bg' +'/10_net_G.pth'
    Tensor2Sketch.load_state_dict(torch.load(sketch_gen_checkpoint))
    Img2Tensor = networks_tensor.define_G(3, 3, 64, 'img2tensor', 
                                  4, 6, 1, 
                                  3, 'instance', gpu_ids)
    image_gen_checkpoint = '../code/face_getsketch_part/checkpoints/img2tensor_bg' +'/10_net_G.pth'
    Img2Tensor.load_state_dict(torch.load(image_gen_checkpoint))

    #img_sketch_path = '2021_03_10_12_13_00_256.jpg'
    img_sketch_path = '00f4558d4bc8f624bcacfe974bb046812ddbe443.jpg'
    sketch_img = read_img(img_sketch_path)

    tensor = Img2Tensor(sketch_img)
    _, sketch = Tensor2Sketch(tensor)
    path = "./"
    name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')  
    utils.save_image(
        sketch,
        path + name + ".png",
        nrow=int(sketch.shape[0] ** 0.5),
        normalize=True,
        range=(-1, 1),
    )