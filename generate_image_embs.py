import sys
sys.path.append('../')
from libs.dataloader import BOLD5000ImageNet
from libs.models import alexnet, AlexNet
from collections import OrderedDict
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch
import os
import rsatoolbox
import pickle

sparsity=0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
AlexNet.forward=forward
model=alexnet(sparsity=sparsity,num_classes=200)
model=model.eval()
model=model.to(device=device)


#load checkpoint

ckpt_dir='/project/6029407/subho/CMPUT652/Effect-of-Sparsity-in-Explaining-IT-VIsual-Cortex/trained_models/sparsity=%s, alexnet/checkpoints/epoch=89-step=750059.ckpt'%sparsity

# 
model_weights=torch.load(ckpt_dir)
modified_state_dict=OrderedDict()
for k, v in model_weights['state_dict'].items():
    modified_state_dict[k.replace('model.','')]=v
    
model.load_state_dict(modified_state_dict)

csis=['CSI1','CSI2','CSI3','CSI4','CSI5']

class BOLD5000ProcessedDataset(Dataset):
    def __init__(self,dataset_dir,csi):
        self.dataset_dir=os.path.join(dataset_dir,csi,'processed')
        self.N=1916
        
    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        image_file=os.path.join(self.dataset_dir,'%d_image.pt'%idx)
        image_tensor=torch.load(image_file)
        return image_tensor

dataset_dir='/home/subho/scratch/Dataset/'
save_dir='/home/subho/scratch/embs/sparsity%s'%sparsity
if os.path.exists(save_dir)==False:
    os.makedirs(save_dir)
dataset=BOLD5000ProcessedDataset(dataset_dir,csis[0])

import time
import tqdm

for i in tqdm.tqdm(range(len(dataset))):
    img=dataset[i]
    img=img.to(device=device)
    img_emb=model(img.unsqueeze(0))
    
    save_path=os.path.join(save_dir,save_dir,'%d_emb.pt'%i)
    torch.save(img_emb,save_path)