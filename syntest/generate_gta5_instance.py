import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from PIL import ImageOps
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as F

import time
from GTA5_dataset import GTA5
import json
import cv2

here = os.getcwd()
data_path = os.path.join(here,'./data/gta5/')
os.makedirs(os.path.join(here,'./data/gta5_instance/'), exist_ok=True)
output_dir = os.path.join(here,'./data/gta5_instance/')
save_phase = 'GTA5'

GTA5_dataset = GTA5(data_path)

class_list = ['unlabeled','ego vehicle','rectification border','out of roi',
'static','dynamic','ground','road','sidewalk','parking','rail track','building','wall',
'fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign',
'vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer',
'train','motorcycle','bicycle']

for i in range(len(GTA5_dataset)):
    sample, label = GTA5_dataset[i]
    sample = np.array(sample)
    _,H,W = sample.shape
    image = Image.fromarray(sample.transpose(1,2,0))

    label = np.array(label)
    classes = np.unique(label)
    savedir = os.path.join(output_dir, save_phase)
    
    label_dict = {}
    os.makedirs(os.path.join(savedir, "label"), exist_ok=True)
    label_file = os.path.join(os.path.join(savedir, "label"), "{}.json".format(i))
    for idx, id in enumerate(classes):
        mask = label == id

        binary = np.asarray(mask, dtype="uint8")
        binary = binary*255
        thresh = 100
        #get threshold image
        ret,thresh_img = cv2.threshold(binary, thresh, 255, cv2.THRESH_BINARY)
        #find contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []

        for contour in contours:
            contour = np.squeeze(contour)
            contour = np.array(contour)
            if(len(contour.shape) != 2):
                #print("error occured!")
                continue

            x_min = np.min(contour[:,0]) - 5
            x_max = np.max(contour[:,0]) + 5

            y_min = np.min(contour[:,1]) - 5
            y_max = np.max(contour[:,1]) + 5
        
            if((x_max-x_min)*(y_max-y_min) < 32*32):
                continue
            if((x_max-x_min) > W or (y_max-y_min) > H):
                continue
            bboxes.append(((x_min,y_min,x_max,y_max)))

        os.makedirs(savedir, exist_ok=True)
        if id > 33:
            continue
        else:
            os.makedirs(os.path.join(savedir, class_list[id]), exist_ok=True)
        # print("Directory structure prepared at %s" % output_dir)
        for bbox_idx,bbox in enumerate(bboxes):
            img = image.crop(bbox)
            savepath = os.path.join(os.path.join(savedir, class_list[id]), "{}_{}_{}.png".format(i,idx,bbox_idx))
            # do not resize to 256 here, do it in the dataloader
            #img = img.convert('RGB').resize((256, 256))
            img.save(savepath, format='png', subsampling=0, quality=100)

            # image.thumbnail((256, 256), Image.ANTIALIAS)
            w_r,h_r = image.size
            # image = ImageOps.pad(image,size=(256, 256),centering=(0, 0))
            
            label_dict.update({str(id)+"_"+str(bbox_idx):list([bbox[0].tolist(),bbox[1].tolist(),bbox[2].tolist(),bbox[3].tolist(),w_r,h_r])})

    with open(label_file, 'w') as json_file:
        json.dump(label_dict, json_file)

    # if i == 20:
    #     break