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

here = os.getcwd()
data_path = os.path.join(here,'./data/GTA5/')
os.makedirs(os.path.join(here,'./originoutput/'), exist_ok=True)
output_dir = os.path.join(here,'./originoutput/')
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

    label = np.array(label)
    classes = np.unique(label)
    savedir = os.path.join(output_dir, save_phase)
    
    label_dict = {}
    os.makedirs(os.path.join(savedir, "label"), exist_ok=True)
    label_file = os.path.join(os.path.join(savedir, "label"), "{}.json".format(i))
    for idx, id in enumerate(classes):
        mask = label == id

        R = np.where(label==id,sample[0,:,:],0)
        G = np.where(label==id,sample[1,:,:],0)
        B = np.where(label==id,sample[2,:,:],0)


        pos = np.where(mask)

        x_min = np.min(pos[1])
        x_max = np.max(pos[1])

        y_min = np.min(pos[0])
        y_max = np.max(pos[0])

        if x_min == x_max or y_min == y_max:
            continue

        img = np.stack([R,G,B])

        image = Image.fromarray(img.transpose(1,2,0))
        image = image.crop((x_min,y_min,x_max,y_max))

        
        os.makedirs(savedir, exist_ok=True)
        if id > 33:
            continue
        else:
            os.makedirs(os.path.join(savedir, class_list[id]), exist_ok=True)
        # print("Directory structure prepared at %s" % output_dir)
        savepath = os.path.join(os.path.join(savedir, class_list[id]), "{}_{}.png".format(i,idx))
        image = image.convert('RGB').resize((256, 256))

        # image.thumbnail((256, 256), Image.ANTIALIAS)
        w_r,h_r = image.size
        # image = ImageOps.pad(image,size=(256, 256),centering=(0, 0))
        image.save(savepath, format='png', subsampling=0, quality=100)
        label_dict.update({str(id):list([x_min.tolist(),y_min.tolist(),x_max.tolist(),y_max.tolist(),w_r,h_r])})

    with open(label_file, 'w') as json_file:
        json.dump(label_dict, json_file)

    if i == 20:
        break