import cv2
import kornia as K
import kornia.feature as KF
from kornia.feature.loftr import LoFTR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import glob
import random

from kornia_moons.feature import *
from PIL import Image

def plot_images(ims):
    
    fig, axes = plt.subplots(3, 3, figsize=(20,20))
    
    for idx, img in enumerate(ims):
        i = idx % 3 
        j = idx // 3 
        image = Image.open(img)
        image = image.resize((300,300))
        axes[i, j].imshow(image)
        axes[i, j].set_title(img.split('/')[-1])

    plt.subplots_adjust(wspace=0, hspace=.2)
    plt.show()
    
def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

def match_and_draw(img_in1, img_in2):
    img1 = load_torch_image(img_in1)
    img2 = load_torch_image(img_in2)
    matcher = LoFTR(pretrained='outdoor')
    
    input_dict = {"image0": K.color.rgb_to_grayscale(img1), 
                  "image1": K.color.rgb_to_grayscale(img2)}
    
    with torch.no_grad():
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0
    
    draw_LAF_matches(
    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
    torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={'inlier_color': (0.2, 1, 0.2),
               'tentative_color': None, 
               'feature_color': (0.2, 0.5, 1), 'vertical': False})
    return correspondences

def plot_matching(samples, files):
    for i in range(samples.shape[1]):
        image_1 = files[samples[0][i]]
        image_2 = files[samples[1][i]]
        print(f'Matching: {image_1} to {image_2}')
        correspondences = match_and_draw(image_1, image_2)


brandenburg_gate_path =  '../input/image-matching-challenge-2022/train/brandenburg_gate/images/'
brandenburg_gate_files = [file for file in glob.glob(f'{brandenburg_gate_path}*.jpg')]

plot_images(random.sample(brandenburg_gate_files, 9))