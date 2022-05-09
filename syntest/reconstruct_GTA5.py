import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image,ImageFilter
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as F

import time
from GTA5_dataset import GTA5
import json
from pathlib import Path

here = os.getcwd()
data_path = os.path.join(here,'./data/GTA5/')
output_dir = os.path.join(here,'./testoutput/')
save_phase = 'GTA5'

GTA5_dataset = GTA5(data_path)

class_list = ['unlabeled','ego vehicle','rectification border','out of roi',
'static','dynamic','ground','road','sidewalk','parking','rail track','building','wall',
'fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign',
'vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer',
'train','motorcycle','bicycle']
savedir = os.path.join(output_dir, save_phase)
label_dir = os.path.join(savedir, "label")
label_files = os.listdir(label_dir)
os.makedirs(os.path.join(savedir, "reconstruct"),exist_ok=True)
print(label_files)

for file in label_files:
    img_idx = file.split(".")[0]
    file_path = os.path.join(label_dir,file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    img = Image.new('RGB', (1914, 1052))
    for key, item in data.items():
        class_idx = class_list[int(key)]
        patch_path = Path(os.path.join(savedir,class_idx))
        PREFIX = img_idx+"_"
        patch_path = [path for path in patch_path.glob(f"{PREFIX}*")]
        patch_path = str(patch_path[0])
        patch_img = Image.open(patch_path).convert('RGB')
        # patch_img = patch_img.crop((0,0,item[4],item[5]))
        # patch_img = patch_img.resize((item[2]-item[0],item[3]-item[1]),resample=Image.LANCZOS)
        patch_img = patch_img.resize((item[2]-item[0],item[3]-item[1]))
        mask1 = np.array(patch_img)[:,:,0] == 0
        mask2 = np.array(patch_img)[:,:,1] == 0
        mask3 = np.array(patch_img)[:,:,2] == 0
        mask = ~mask1 + ~mask2 + ~mask3
        mask = Image.fromarray(mask)
        img.paste(patch_img, (item[0], item[1]),mask=mask)

    # img = img.filter(ImageFilter.BoxBlur(5))
    print(len(data.keys()))
    savepath = os.path.join(os.path.join(savedir, "reconstruct"), "{}.png".format(img_idx))
    img.save(savepath, format='png', subsampling=0, quality=100)

# for i in range(len(GTA5_dataset)):
#     sample, label = GTA5_dataset[i]
#     sample = np.array(sample)

#     label = np.array(label)
#     classes = np.unique(label)
#     savedir = os.path.join(output_dir, save_phase)
    
#     label_dict = {}
#     os.makedirs(os.path.join(savedir, "label"), exist_ok=True)
#     label_file = os.path.join(os.path.join(savedir, "label"), "{}.json".format(i))
#     for idx, id in enumerate(classes):
#         mask = label == id

#         R = np.where(label==id,sample[0,:,:],0)
#         G = np.where(label==id,sample[1,:,:],0)
#         B = np.where(label==id,sample[2,:,:],0)


#         pos = np.where(mask)

#         x_min = np.min(pos[1])
#         x_max = np.max(pos[1])

#         y_min = np.min(pos[0])
#         y_max = np.max(pos[0])

#         if x_min == x_max or y_min == y_max:
#             continue

#         img = np.stack([R,G,B])

#         image = Image.fromarray(img.transpose(1,2,0))
#         image = image.crop((x_min,y_min,x_max,y_max))

        
#         os.makedirs(savedir, exist_ok=True)
#         if id > 33:
#             continue
#         else:
#             os.makedirs(os.path.join(savedir, class_list[id]), exist_ok=True)
#         # print("Directory structure prepared at %s" % output_dir)
#         savepath = os.path.join(os.path.join(savedir, class_list[id]), "{}_{}.png".format(i,idx))
#         image = image.convert('RGB').resize((256, 256))
#         image.save(savepath, format='png', subsampling=0, quality=100)
#         label_dict.update({str(id):list([x_min.tolist(),y_min.tolist(),x_max.tolist(),y_max.tolist()])})

#     with open(label_file, 'w') as json_file:
#         json.dump(label_dict, json_file)