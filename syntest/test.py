from matplotlib.transforms import Bbox
import torch
from torch import float32, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as F

import time
from GTA5_dataset import GTA5
from RealSyntheticmixDataset import GTA2Cityscapes
from Synthnia import Synthnia
from torchvision.datasets import cityscapes

def kitti_collate(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    images, labels = zip(*data)

    imgs = []
    for img in images:
        img = np.array(img)
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        img = F.crop(img,0,0,370,1240)
        imgs.append(img)

    images = torch.stack(imgs,dim=0)

    return images,labels

here = os.getcwd()
root_path = os.path.join(here,"./output")
class_list = ['unlabeled','ego vehicle','rectification border','out of roi',
'static','dynamic','ground','road','sidewalk','parking','rail track','building','wall',
'fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign',
'vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer',
'train','motorcycle','bicycle']

# GC = GTA2Cityscapes(root=root_path,class_list=class_list)
# print(len(GC))

# synthnia_path = os.path.join(here,"data/Synthnia/")
# synthnia_dataset = Synthnia(synthnia_path)

# sample,label = synthnia_dataset[300]
# print(sample.shape)
# print(label.shape)
# sample1 = np.array(sample).transpose(1,2,0)
# image = Image.fromarray(sample1)
# image.save("1.png")
# label = np.array(label)[:,:,1]
# print(np.unique(label))
# id = 3
# print(sample.shape)
# mask = label == id
# R = np.where(label==id,sample[0,:,:],0)
# G = np.where(label==id,sample[1,:,:],0)
# B = np.where(label==id,sample[2,:,:],0)
# print(R.shape)
# print(G.shape)
# print(B.shape)


# pos = np.where(mask)
# x_min = np.min(pos[1])
# x_max = np.max(pos[1])

# y_min = np.min(pos[0])
# y_max = np.max(pos[0])
# img = np.stack([R,G,B])
# print(img.shape)

# image = Image.fromarray(img.transpose(1,2,0))
# image = image.crop((x_min,y_min,x_max,y_max))

# image.save("2.png")
# kitti_path = os.path.join(here,'./data/')

# kitti_dataset = torchvision.datasets.Kitti(root=kitti_path,download=False)

data_path = os.path.join(here,'./data/gta5/')

GTA5_dataset = GTA5(data_path)
print(len(GTA5_dataset))

sample, label = GTA5_dataset[55]
sample = np.array(sample)
image = Image.fromarray(sample.transpose(1,2,0))

image.save("1.png")
label = np.array(label)
_,H,W = sample.shape
print(sample.shape)
print(np.unique(label))

id = 24
mask = label == id
import cv2
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
        print("error occured!")
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
print(bboxes)



# R = np.where(label==id,sample[0,:,:],0)
# G = np.where(label==id,sample[1,:,:],0)
# B = np.where(label==id,sample[2,:,:],0)
# print(R.shape)
# print(G.shape)
# print(B.shape)


# pos = np.where(mask)
# x_min = np.min(pos[1])
# x_max = np.max(pos[1])

# y_min = np.min(pos[0])
# y_max = np.max(pos[0])
# img = np.stack([R,G,B])

# image = Image.fromarray(img.transpose(1,2,0))
for idx,bbox in enumerate(bboxes):
    img = image.crop(bbox)
    img.save("{}.png".format(idx+3))
# image = image.crop((x_min,y_min,x_max,y_max))
# image.save("2.png")

# batch_size = 1
# mixed_synth = 0.4
# real_dataloader = DataLoader(kitti_dataset,batch_size=batch_size,
#                         shuffle=True, num_workers=0,collate_fn=kitti_collate)

# synth_dataloader = DataLoader(GTA5_dataset,batch_size=batch_size,
#                         shuffle=True, num_workers=0)

# for iter, (real,synth) in enumerate(zip(real_dataloader, synth_dataloader)):
#     print(real)
#     print(synth)
# import torchvision.transforms as transforms

# transform = transforms.Compose([
#     # you can add other transformations in this list
#     transforms.ToTensor()
# ])
# cityscapes_path = os.path.join(here,'./data/cityscapes/')
# cityscapes_dataset = torchvision.datasets.Cityscapes(root=cityscapes_path,split="train",mode="fine",target_type=["semantic","color"])
# sample,label = cityscapes_dataset[20]
# # img = Image.fromarray(np.array(label[1]))
# # img.save("1.png")

# semantic = np.array(label[0])
# color = np.array(label[1])
# color[:,:,3] = semantic
# mask = color.transpose(2,0,1)
# mask = mask[:3,:,:]
# print(mask.shape)
# # print(np.array(label).shape)

# # instances are encoded as different colors
# obj_ids = np.unique(mask,axis=0)
# # first id is the background, so remove it
# obj_ids = obj_ids[1:]


# # split the color-encoded mask into a set
# # of binary masks
# masks = mask == obj_ids[:, None, None]


# # get bounding box coordinates for each mask
# num_objs = len(obj_ids)
# boxes = []
# for i in range(num_objs):
#     pos = np.where(masks[i])
#     xmin = np.min(pos[1])
#     xmax = np.max(pos[1])
#     ymin = np.min(pos[0])
#     ymax = np.max(pos[0])
#     boxes.append([xmin, ymin, xmax, ymax])

# print(boxes)

# cityscapes_dataloader = DataLoader(cityscapes_dataset,batch_size=batch_size,
#                         shuffle=True, num_workers=0)
# for iter, (sample,label) in enumerate(cityscapes_dataloader):
#     if iter ==10:
#         print(label[0,:,:,:])
#         print(np.unique(label[0,0,:,:].numpy()))