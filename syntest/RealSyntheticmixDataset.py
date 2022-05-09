import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset
import os
from pathlib import Path


class GTA2Cityscapes(torchDataset):
    def __init__(self, root, class_list) -> None:
        super().__init__()

        self.root = root
        self.class_list = class_list

        self.GTA5_path = os.path.join(self.root,'GTA5')
        self.cityscape_path = os.path.join(self.root,'cityscapes')
        self.SUFFIX = ".png"

        sample = []
        dirs_GTA = os.listdir(self.GTA5_path)
        dirs_cityscapes = os.listdir(self.cityscape_path)
        for idx in class_list:
            if dirs_GTA.count(idx) >0 and dirs_cityscapes.count(idx) >0:
                print("both dataset has {}".format(idx))

                class_dir_GTA = Path(os.path.join(self.GTA5_path, idx))
                class_dir_cityscapes = Path(os.path.join(self.cityscape_path, idx))
                img_path_GTA = [path for path in class_dir_GTA.glob(f"*{self.SUFFIX}")]
                img_path_cityscapes = [path for path in class_dir_cityscapes.glob(f"*{self.SUFFIX}")]

                if len(img_path_GTA) > len(img_path_cityscapes):
                    img_path_GTA = img_path_GTA[:len(img_path_cityscapes)]
                else:
                    img_path_cityscapes = img_path_cityscapes[:len(img_path_GTA)]

                for idx,(G,C) in enumerate(zip(img_path_GTA,img_path_cityscapes)): 
                    sample.append((G,C))
                
        self.sample = sample
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        G,C = self.sample[index]
        G = Image.open(str(G))
        G = np.array(G).transpose(2,0,1)
        C = Image.open(str(C))
        C = np.array(C).transpose(2,0,1)
        return G,C


class cat_dataloaders():
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise StopIteration
        return tuple(out)