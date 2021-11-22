import sys
sys.path.append('../')
from libs.dataloader import BOLD5000ImageNet
from pydicom import dcmread
import tqdm
import torch
import os
import pickle

for csi in range(4,5):
    fmri_data_path='/home/subho/scratch/Dataset/CSI%d'%csi
    fmri_dataset=BOLD5000ImageNet(fmri_data_path)
    images=[]
    fmris=[]
    for i in tqdm.tqdm(range(len(fmri_dataset))):
        img,fmri=fmri_dataset[i]
        torch.save(img,os.path.join('/home/subho/scratch/Dataset/CSI%d/processed/'%csi,'%d_image.pt'%i))
        torch.save(fmri,os.path.join('/home/subho/scratch/Dataset/CSI%d/processed/'%csi,'%d_fmri.pt'%i))