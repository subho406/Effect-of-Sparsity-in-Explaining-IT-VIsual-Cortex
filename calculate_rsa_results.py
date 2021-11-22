#!/usr/bin/env python
# coding: utf-8

# # Evaluation Notebook
# 
# Compares the similarity of feature representations from a trained dense network with BOLD5000 ImageNet fMRI samples

# In[26]:

import json
import sys
import numpy as np
import torch.nn as nn
sys.path.append('../')
from libs.dataloader import BOLD5000ImageNet
from libs.models import alexnet, AlexNet
from collections import OrderedDict
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch
import os
from sklearn.linear_model import SGDRegressor, Ridge, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from neurora.rdm_cal import bhvRDM
import pickle
import rsatoolbox



# In[27]:



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### Load the Image embs and FMRI embeddings

# In[68]:


img_emb_dir='/home/subho/scratch/embs/sparsity%s'
fmri_emb_dir='/home/subho/scratch/Dataset/%s/processed'


# In[70]:


class ProcessedFMRI(Dataset):
    def __init__(self,fmri_emb_dir):
        self.fmri_emb_dir=fmri_emb_dir
    def __len__(self):
        return 1916
    def __getitem__(self,idx):
        fmri=torch.load(os.path.join(self.fmri_emb_dir,'%d_fmri.pt'%idx),map_location='cpu')
        return fmri


# In[57]:


class ProcessedEMB(Dataset):
    def __init__(self,img_emb_dir):
        self.img_emb_dir=img_emb_dir
    def __len__(self):
        return 1916
    def __getitem__(self,idx):
        img=torch.load(os.path.join(self.img_emb_dir,'%d_emb.pt'%idx),map_location='cpu')
        return img


# In[51]:


results=[]


# ### Start with fetching the fmri embs for one subject

# In[66]:


csis=['CSI1','CSI2','CSI3']


# In[71]:

for csi in csis:
    dataset=ProcessedFMRI(fmri_emb_dir%csi)
    fmri_embs=[]

    for i in tqdm.tqdm(range(len(dataset))):
        fmri=dataset[i]
        fmri=(fmri/2047)
        fmri_embs.append(fmri)
    fmri_embs=np.stack(fmri_embs,axis=0)
    fmri_embs=fmri_embs.reshape(1916,-1)


    # In[72]:


    fmri_data=rsatoolbox.data.Dataset(fmri_embs)
    fmri_rdm = rsatoolbox.rdm.calc_rdm(fmri_data)
    #Calculate the FMRI RDM 


    # In[ ]:


    sparsity_values=[0,0.2,0.4,0.6,0.8]
    for sparsity in sparsity_values:
        print("Calculating for Sparsity: %s"%sparsity)
        dataset=ProcessedEMB(img_emb_dir%sparsity)
        img_embs=[]
        for i in tqdm.tqdm(range(len(dataset))):
            img=dataset[i]
            img_embs.append(img.detach().numpy())
        img_embs=np.concatenate(img_embs,axis=0)
        img_data=rsatoolbox.data.Dataset(img_embs)
        img_rdm = rsatoolbox.rdm.calc_rdm(img_data)
        comparison_pearson=rsatoolbox.rdm.compare(img_rdm,fmri_rdm)
        comparison_kendall=rsatoolbox.rdm.compare(img_rdm,fmri_rdm,method='kendall')
        print(comparison_pearson,comparison_kendall)
        results.append({'subject':csi,'sparsity':sparsity,'rdm-pearson':comparison_pearson,'rdm-kendall':comparison_kendall})

print(results)
with open('/home/subho/projects/def-afyshe-ab/subho/CMPUT652/Effect-of-Sparsity-in-Explaining-IT-VIsual-Cortex/rsa_results.txt','w') as f:
    f.write(str(results))