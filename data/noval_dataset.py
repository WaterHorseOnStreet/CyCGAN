import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset
import os
from pathlib import Path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


# class NovalDataset(BaseDataset):
#     def __init__(self, opt) -> None:
#         BaseDataset.__init__(self, opt)


#         self.root = opt.dataroot
#         # class_list = ['unlabeled','ego vehicle','rectification border','out of roi',
#         #             'static','dynamic','ground','road','sidewalk','parking','rail track','building','wall',
#         #             'fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign',
#         #             'vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer',
#         #             'train','motorcycle','bicycle']
#         class_list = ["road","building","person","car"]
#         # class_list = ["person"]
#         self.class_list = class_list

#         self.GTA5_path = os.path.join(self.root,'GTA5')
#         self.cityscape_path = os.path.join(self.root,'cityscapes')
#         self.SUFFIX = ".png"

#         sample = []
#         dirs_GTA = os.listdir(self.GTA5_path)
#         dirs_cityscapes = os.listdir(self.cityscape_path)
#         A_path = []
#         B_path = []
#         for idx in class_list:
#             if dirs_GTA.count(idx) >0 and dirs_cityscapes.count(idx) >0:
#                 print("both dataset has {}".format(idx))

#                 class_dir_GTA = Path(os.path.join(self.GTA5_path, idx))
#                 class_dir_cityscapes = Path(os.path.join(self.cityscape_path, idx))
#                 img_path_GTA = [path for path in class_dir_GTA.glob(f"*{self.SUFFIX}")]
#                 img_path_cityscapes = [path for path in class_dir_cityscapes.glob(f"*{self.SUFFIX}")]

#                 if len(img_path_GTA) > len(img_path_cityscapes):
#                     img_path_GTA = img_path_GTA[:len(img_path_cityscapes)]
#                 else:
#                     img_path_cityscapes = img_path_cityscapes[:len(img_path_GTA)]

#                 for idx,(G,C) in enumerate(zip(img_path_GTA,img_path_cityscapes)): 
#                     A_path.append(str(G))
#                     B_path.append(str(C))
#                     sample.append((str(G),str(C)))
                
#         self.sample = sample
#         self.A_path = A_path
#         self.B_path = B_path

#         self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
#         self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
    
#     def __len__(self):
#         return len(self.sample)

#     def __getitem__(self, index):
#         A,B = self.sample[index]
#         A = Image.open(str(A)).convert('RGB')
#         B = Image.open(str(B)).convert('RGB')

#         transform_params = get_params(self.opt, A.size)
#         A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
#         B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

#         A = A_transform(A)
#         B = B_transform(B)

#         return {'A': A, 'B': B, 'A_paths': self.sample[index][0], 'B_paths': self.sample[index][1]}

class NovalDataset(BaseDataset):
    def __init__(self, opt) -> None:
        BaseDataset.__init__(self, opt)


        self.root = opt.dataroot

        self.GTA5_path = os.path.join(self.root,'gta5/images')
        self.cityscape_path = os.path.join(self.root,'cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/')
        self.SUFFIX = ".png"

        sample = []
        dirs_GTA = Path(self.GTA5_path)
        # dirs_cityscapes = Path(self.cityscape_path)
        dirs_cityscapes = os.listdir(self.cityscape_path)
        A_path = []
        B_path = []
        img_path_cityscapes = []
        img_path_GTA = [path for path in dirs_GTA.glob(f"*{self.SUFFIX}")]
        for folder in dirs_cityscapes:
            subfolders = Path(os.path.join(self.cityscape_path,folder))
            img_path_cityscapes = img_path_cityscapes + [path for path in subfolders.glob(f"*{self.SUFFIX}")]

        if len(img_path_GTA) > len(img_path_cityscapes):
            img_path_GTA = img_path_GTA[:len(img_path_cityscapes)]
        else:
            img_path_cityscapes = img_path_cityscapes[:len(img_path_GTA)]

        for idx,(G,C) in enumerate(zip(img_path_GTA,img_path_cityscapes)): 
            A_path.append(str(G))
            B_path.append(str(C))
            sample.append((str(G),str(C)))
                
        self.sample = sample
        self.A_path = A_path
        self.B_path = B_path

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
    
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        A,B = self.sample[index]
        A = Image.open(str(A)).convert('RGB')
        B = Image.open(str(B)).convert('RGB')

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': self.sample[index][0], 'B_paths': self.sample[index][1]}
