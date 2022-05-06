from sklearn.feature_extraction import image


import os
from pathlib import Path
from datasets.GTA5_dataset import GTA5
import numpy as np
from PIL import Image

here = os.getcwd()
GTA5_path = os.path.join(here,'../syntest/data/GTA5/')
fake_dict = Path(os.path.join(here,"./results/maps_cyclegan/test_latest/images"))
PREFIX = "fake_B.png"
fake_path = [path for path in fake_dict.glob(f"*{PREFIX}")]
class_list = ["road","building","person","car"]
class_list = ['unlabeled','ego vehicle','rectification border','out of roi',
'static','dynamic','ground','road','sidewalk','parking','rail track','building','wall',
'fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign',
'vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer',
'train','motorcycle','bicycle']


GTA5_dataset = GTA5(GTA5_path)
for path in fake_path:
    path = str(path)
    img_name = path.split('/')[-1]
    img_name = img_name.split('.')[0]
    img_idx = int(img_name.split('_')[0])
    class_idx = int(img_name.split('_')[1])

    _, label = GTA5_dataset[img_idx]

    label = np.array(label)
    classes = np.unique(label)

    class_name = class_list[classes[class_idx]]

    os.makedirs(os.path.join(here,class_name),exist_ok=True)

    save_path = os.path.join(os.path.join(here,class_name),'{}_{}.png'.format(img_idx,class_idx))
    image = Image.open(path)
    image.save(save_path, format='png', subsampling=0, quality=100)