import os
from torchvision.io import read_image

from distutils.dir_util import copy_tree
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread
from PIL import Image
import torchvision.transforms as transforms
import shutil
import torch
import time
import numpy as np

class BOLD5000ImageNet(Dataset):
    def __init__(self , bold_data_dir):
        self.bold_data_dir=bold_data_dir
        self.images = os.listdir(os.path.join(bold_data_dir,'images'))
        self.fmris = os.listdir(os.path.join(bold_data_dir,'dcm_files'))
        self.dataset_size = len(self.images)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms=transforms.Compose(
                    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
                )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img=Image.open(os.path.join(self.bold_data_dir,'images',self.images[idx]))
        img_tensor=self.transforms(img)
        fmri_folder=os.path.join(self.bold_data_dir,'dcm_files',self.fmris[idx])
        fmri_file_list=sorted(os.listdir(fmri_folder))
        pixel_arrays=[]
        for f in fmri_file_list:
            dcm=dcmread(os.path.join(fmri_folder,f))
            pixel_arrays.append(dcm.pixel_array.copy())
        pixel_arrays=torch.tensor(np.stack(pixel_arrays,axis=0).astype(int),dtype=torch.int16) #Convert to float by dividing by max value 2047 (11 bit integer)
        return img_tensor,pixel_arrays