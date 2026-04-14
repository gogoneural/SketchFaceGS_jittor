import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.onnx
import torchvision.transforms as transforms

import cv2
import os
from PIL import Image
from scipy.io import loadmat
from skimage.transform import estimate_transform, warp, resize, rescale
from kornia.geometry.transform import warp_perspective, get_perspective_transform
import gghead_jittor.face_utils.box_utils_numpy as box_utils
from gghead_jittor.env import REPO_ROOT_DIR
from roma import rotvec_to_rotmat

# landmark detection setting
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])
resize = transforms.Resize([56, 56])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# landmark detection model
import onnxruntime


class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # scale to [0,1]
    def projectLandmark(self, landmark):
        return (landmark - np.array([self.x, self.y]))/np.array([self.w, self.h])

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        return landmark * np.array([self.w, self.h]) + np.array([self.x, self.y])


class FLAMERecon():
    def __init__(self, device='cuda:0'):
        self.crop = CropFace(device=device)
        # FLAME model setting
        self.param_dict = {'shape': 100, 'tex': 50, 'exp': 50, 'pose': 6, 'cam': 3, 'light': 27}
        # deca model
        deca_onnx_path = os.path.join(REPO_ROOT_DIR, "assets", "deca.onnx")
        self.ort_session_deca = onnxruntime.InferenceSession(deca_onnx_path, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': self.crop.device_id}])
        self.deca_input_name = self.ort_session_deca.get_inputs()[0].name

    def reconstruct_FLAME_from_rawimg(self, image):
        '''
        img: (H, W, C), BGR, [0, 255]
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        return self.reconstruct_FLAME(image)
    
    def reconstruct_FLAME(self, images):
        '''
        images: (B, C, H, W), RGB, [-1, 1]
        '''
        images = self.crop(images)
        images = (images.cpu().numpy().astype(np.float32) + 1) / 2
        code = self.ort_session_deca.run(None, {self.deca_input_name: images})[0]
        code_dict = {}
        start = 0
        for key in self.param_dict:
            end = start + self.param_dict[key]
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(-1, 9,3)
            code_dict[key] = code_dict[key].tolist()
        return code_dict

class CropFace(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.device_id = self.device.index
        super(CropFace, self).__init__()
        landmark_onnx_path = os.path.join(REPO_ROOT_DIR, "assets", "landmark_detection_56_se_external.onnx")
        detection_onnx_path = os.path.join(REPO_ROOT_DIR, "assets", "version-RFB-320.onnx")
        self.ort_session_landmark = onnxruntime.InferenceSession(landmark_onnx_path, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': self.device_id}])
        self.landmark_input_name = self.ort_session_landmark.get_inputs()[0].name
        self.ort_session_detection = onnxruntime.InferenceSession(detection_onnx_path, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': self.device_id}])
        self.detection_input_name = self.ort_session_detection.get_inputs()[0].name

    
    def forward(self, images):
        '''
        images: (B, C, H, W), RGB, [-1, 1]
        '''
        with torch.no_grad():
            kpts = self.predict_landmarks68(images)
            left = np.min(kpts[:,:,0], axis=1); right = np.max(kpts[:,:,0], axis=1)
            top = np.min(kpts[:,:,1], axis=1); bottom = np.max(kpts[:,:,1], axis=1)
            old_size = (right - left + bottom - top)/2*1.1
            center = np.stack([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ], axis=-1)
            size = (old_size*1.25).astype(np.int32).reshape(-1,1).repeat(2, axis=1)
            src_pts = np.stack([center + size/2*np.array([-1, -1]),
                                      center + size/2*np.array([-1, 1]),
                                      center + size/2*np.array([1, -1]),
                                      center + size/2*np.array([1, 1])], axis=1).astype(np.float32)
            DST_PTS = np.array([[[0,0], [0,224 - 1], [224 - 1, 0], [224 - 1, 224 - 1]]]).repeat(images.size(0), axis=0).astype(np.float32)
            t = get_perspective_transform(torch.tensor(src_pts).to(images.device), torch.tensor(DST_PTS).to(images.device))
        
        dst_images = warp_perspective(images, t, dsize=(224, 224))
        return dst_images

    def predict_landmarks68(self, images):
        height, width = images.shape[-2:]
        img_input = F.interpolate(images, size=(240, 320), mode='bilinear', align_corners=True)
        confidences, boxes = self.ort_session_detection.run(None, {self.detection_input_name: img_input.cpu().numpy().astype(np.float32)})
        boxes = predict(width, height, confidences, boxes)
        out_size = 56
        x1=boxes[:, 0]
        y1=boxes[:, 1]
        x2=boxes[:, 2]
        y2=boxes[:, 3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = np.maximum(w, h)*1.1
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size
        dx = np.maximum(0, -x1)
        dy = np.maximum(0, -y1)
        x1 = np.maximum(0, x1)
        y1 = np.maximum(0, y1)

        edx = np.maximum(0, x2 - width)
        edy = np.maximum(0, y2 - height)
        x2 = np.minimum(width, x2)
        y2 = np.minimum(height, y2)
        boxes = np.stack([x1, x2, y1, y2], axis=1).astype(np.int32)
        cropped_faces = []
        landmarks = []
        for bboxi, dxi, dyi, edxi, edyi, img in zip(boxes, dx, dy, edx, edy, images):
            new_bbox = BBox(bboxi)
            cropped=img[:, new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dxi > 0 or dyi > 0 or edxi > 0 or edyi > 0):
                cropped = F.pad(cropped, (int(dyi), int(edyi), int(dxi), int(edxi)), mode='constant', value=-1)
            cropped_face = F.interpolate(cropped.unsqueeze(0), size=(out_size, out_size), mode='bilinear', align_corners=True)
            cropped_face = (cropped_face + 1)/2
            cropped_face = normalize(cropped_face)
            ort_inputs = {self.landmark_input_name: cropped_face.detach().cpu().numpy()}
            landmark = self.ort_session_landmark.run(None, ort_inputs)[0]
            landmark = landmark.reshape(68,2)
            landmarks.append(new_bbox.reprojectLandmark(landmark))
        return np.stack(landmarks)

def predict(width, height, confidences, boxes):
    class_index = 1
    probs = confidences[:, :, class_index]
    boxes = boxes[range(boxes.shape[0]), probs.argmax(1)]
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    return boxes

def angle2cam(angle):
    R = rotvec_to_rotmat(angle[:, :3])
    t = torch.zeros_like(angle[:, :3])
    t[:, 2] = -10
    c = - R.transpose(1, 2) @ t.unsqueeze(-1)
    R = torch.linalg.inv(R)
    pose = torch.eye(4, device=angle.device).unsqueeze(0).repeat(angle.size(0), 1, 1)
    pose[:, :3, :3] = R
    
    c *= 0.27
    c[:, 1] += 0.006
    c[:, 2] += 0.161
    pose[:, :3, 3] = c.squeeze(-1)

    Rot = torch.eye(3, device=angle.device).unsqueeze(0)
    Rot[:, 0, 0] = 1
    Rot[:, 1, 1] = -1
    Rot[:, 2, 2] = -1
    pose[:, :3, :3] = pose[:, :3, :3] @ Rot
    return pose

if __name__ == "__main__":
    imgs = [cv2.imread('../../ffhq/ffhq_512/img0000000{}.png'.format(i), cv2.IMREAD_UNCHANGED)[..., :3] for i in range(6)]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = np.array(imgs).transpose(0,3,1,2).astype(np.float32)/127.5 - 1
    imgs = torch.from_numpy(imgs).to(device)
    crop = CropFace(device=device)
    dst_imgs = crop(imgs)
    dst_imgs = dst_imgs.cpu().add(1).mul(127.5).byte().numpy().transpose(0,2,3,1)
    for i, img in enumerate(dst_imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"wasted/face_{i}.png", img)
    
    flame = FLAMERecon(device=device)
    code_dict = flame.reconstruct_FLAME(imgs)
    print(code_dict)
    