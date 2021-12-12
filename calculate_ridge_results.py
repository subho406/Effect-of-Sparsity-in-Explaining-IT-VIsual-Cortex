#!/usr/bin/env python
# coding: utf-8

# # Evaluation Notebook
# 
# Compares the similarity of feature representations from a trained dense network with BOLD5000 ImageNet fMRI samples

# In[ ]:


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
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import pickle
import sklearn
from torchvision.transforms  import Resize
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
#from sklearn.preprocessing import StandardScalar


# In[ ]:



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### Load the Image embs and FMRI embeddings

# In[ ]:


img_emb_dir='/home/subho/scratch/embs/sparsity%s'
fmri_emb_dir='/home/subho/scratch/Dataset/%s/processed'


# In[ ]:


class ProcessedFMRI(Dataset):
    def __init__(self,fmri_emb_dir):
        self.fmri_emb_dir=fmri_emb_dir
    def __len__(self):
        return 1916
    def __getitem__(self,idx):
        fmri=torch.load(os.path.join(self.fmri_emb_dir,'%d_fmri.pt'%idx),map_location='cpu')
        return fmri


# In[ ]:


class ProcessedEMB(Dataset):
    def __init__(self,img_emb_dir):
        self.img_emb_dir=img_emb_dir
    def __len__(self):
        return 1916
    def __getitem__(self,idx):
        img=torch.load(os.path.join(self.img_emb_dir,'%d_emb.pt'%idx),map_location='cpu')
        return img


# In[ ]:


results=[]


# ### Start with fetching the fmri embs for one subject

# In[ ]:


def k_fold_linear_regression(fmri_embs,img_embs):
    np.random.seed(49)
    #scaler=StandardScaler()
    #fmri_embs=scaler.fit_transform(fmri_embs)
    input_dim=img_embs.shape[1]
    output_dim=fmri_embs.shape[1]
    
    
    kf=sklearn.model_selection.KFold(10,shuffle=True,random_state=49)
    k_fold_scores_clf=[]
    r2_scores=[]
    for train_index, test_index in tqdm.tqdm(kf.split(img_embs)):
        img_embs_train,img_embs_test=img_embs[train_index],img_embs[test_index]
        fmri_embs_train,fmri_embs_test=fmri_embs[train_index],fmri_embs[test_index]
        model=RidgeCV(normalize=True,alphas=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000],alpha_per_target=True)
        print("Fitting model")
        model.fit(img_embs_train,fmri_embs_train)
       
        #Evaluate on the test set
        #Calculate the regression loss
        img_embs_test_shuffled=img_embs_test.copy()
        img_embs_test_shuffled=img_embs_test_shuffled[np.random.permutation(img_embs_test.shape[0])]
        pred_embs_correct=model.predict(img_embs_test)
        pred_embs_shuffled=model.predict(img_embs_test_shuffled)
        r2_score=model.score(img_embs_test,fmri_embs_test)
        r2_scores.append(r2_score)
        print(r2_score)
        #Calculate the two-choice classification score - Whole brain classification score 
        euclid_correct=euclidean_distances(pred_embs_correct,fmri_embs_test).diagonal()
        euclid_shuffled=euclidean_distances(pred_embs_shuffled,fmri_embs_test).diagonal()
        clf_sum=np.sum(np.stack([euclid_correct,euclid_shuffled],axis=1).argmin(axis=1)==0)
        #print(clf_sum/img_embs_test.shape[0])
        k_fold_scores_clf.append(clf_sum)
    return np.sum(k_fold_scores_clf)/img_embs.shape[0],np.mean(r2_scores)


csis=['CSI1','CSI2','CSI3']


# In[ ]:
results=[]

for csi in csis:
    dataset=ProcessedFMRI(fmri_emb_dir%csi)
    fmri_embs=[]
    resize=Resize(size=(120,120),interpolation=InterpolationMode.NEAREST)
    for i in tqdm.tqdm(range(len(dataset))):
        fmri=dataset[i]
        fmri=resize(fmri)
        fmri=(fmri[2:4]/2047).mean(axis=0)
        fmri_embs.append(fmri)
    fmri_embs=np.stack(fmri_embs,axis=0)
    fmri_embs=fmri_embs.reshape(1916,-1)

    sparsity_values=[0,0.2,0.4,0.6,0.8]
    for sparsity in sparsity_values:
        print("Calculating for Sparsity: %s"%sparsity)
        dataset=ProcessedEMB(img_emb_dir%sparsity)
        img_embs=[]
        for i in tqdm.tqdm(range(len(dataset))):
            img=dataset[i]
            img_embs.append(img.detach().numpy())
        img_embs=np.concatenate(img_embs,axis=0)
        twovstwo,r2=k_fold_linear_regression(fmri_embs,img_embs)
        print('Score found: ',twovstwo,r2)
        results.append({'subject':csi,'sparsity':sparsity,'twovstwo':twovstwo,'r2':r2})

print(results)
with open('/home/subho/projects/def-afyshe-ab/subho/CMPUT652/Effect-of-Sparsity-in-Explaining-IT-VIsual-Cortex/ridge_results.txt','w') as f:
    f.write(str(results))

