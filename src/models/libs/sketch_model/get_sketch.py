import time
import torch
import torch.nn as nn
import functools
import os
import datetime

from torchvision import transforms, utils
from PIL import Image, ImageFilter
from torch.autograd import Variable
import networks_sketch

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
    SketchGen = networks_sketch.define_S(3, 3, 64, "sketch_no_part", 'instance', not False, 'normal', 0.02, gpu_ids)
    
    """
    #img_dir = "../dataset_ffhq/train_img/"
    #img_dir = "/mnt/16T/liufenglin/dataset/FFHQ_recrop/"
    img_dir = "/mnt/4T/liufenglin/dataset/img/"
    #img_dir = "/mnt/16T/liufenglin/dataset_Celeb/test_img/"
    #img_dir = "/mnt/16T/liufenglin/code2/mask_generate/HRNet-landmarks-master/mask_gen/img_swap_mask/"
    #img_dir = "/home/liufenglin/210_16T/dataset/FFHQ_recrop/"
    names = sorted(os.listdir(img_dir))
    #del names[0]
    #result_path = "../dataset_ffhq/train_pred/"
    #result_path = "/mnt/16T/liufenglin/dataset/FFHQ_sketch/"
    result_path = "/mnt/4T/liufenglin/dataset/sketch/"
    #result_path = "/mnt/16T/liufenglin/dataset_Celeb/test_sketch/"
    #result_path = "/mnt/16T/liufenglin/code2/mask_generate/HRNet-landmarks-master/mask_gen/img_swap_mask_sketch/"
    #result_path = "/home/liufenglin/210_16T/dataset/FFHQ_sketch/"
    #for name in names:
    #for i in range(20000):
    #for i in range(20000, 40000):
    #for i in range(40000, 60000):
    #for i in range(60000, 80000):
    #for i in range(80000, 100000):
    #for i in range(100000, 120000):
    #for i in range(130000, 135000):
    #for i in range(130340, 130350):
    #for i in range(15000, 15000 + 7000):
    #for i in range(15000 + 7000, 29000):
    #for i in range(0, 100):
    for i in range(0, 30000):
        name = names[i]
        img_sketch_path = img_dir + name
        sketch_img = read_img(img_sketch_path)
        with torch.no_grad():
            sketch = SketchGen(sketch_img)
        #name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print("\r" + name, end = "")
        utils.save_image(
            sketch,
            result_path + os.path.splitext(name)[0] + ".png",
            nrow=int(sketch.shape[0] ** 0.5),
            normalize=True,
            range=(-1, 1),
        )
    """
    
    
    input_path = './test_sketch/kang.png'
    output_path = './test_sketch/kang_sketch.jpg'
    #input_path = './test_sketch/sketch.jpg'
    #output_path = './test_sketch/sketch_new.jpg'
    sketch_img = read_img(input_path)
    sketch = SketchGen(sketch_img)
    utils.save_image(
        sketch,
        output_path,
        nrow=int(sketch.shape[0] ** 0.5),
        normalize=True,
        range=(-1, 1),
    )
    

