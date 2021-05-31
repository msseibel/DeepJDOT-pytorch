#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:26:41 2018

Paper and original tensorflow implementation: damodara

Pytorch implementation
@author: msseibel

DeepJDOT: with emd for the sample data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import torch
import torch.nn.functional as F
import tqdm

#seed=1985
#np.random.seed(seed)

#%%
source_traindata, source_trainlabel = make_blobs(1200, centers=[[0, -1], [0, 0], [0, 1]], cluster_std=0.2)
target_traindata, target_trainlabel = make_blobs(1200, centers=[[1, 0], [1, 1], [1, 2]], cluster_std=0.2)
plt.figure()
plt.scatter(source_traindata[:,0], source_traindata[:,1], c=source_trainlabel, marker='o', alpha=0.4)
plt.scatter(target_traindata[:,0], target_traindata[:,1], c=target_trainlabel, marker='x', alpha=0.4)
plt.legend(['source train data', 'target train data'])
plt.title("2D blobs visualization (shape=domain, color=class)")

#%% optimizer
n_class = len(np.unique(source_trainlabel))
n_dim = np.shape(source_traindata)

#%% feature extraction and classifier function definition
class BlobModel(torch.nn.Module):
    def __init__(self):
        super(BlobModel, self).__init__()
        self.fc1    = torch.nn.Linear(2,500)
        self.fc2    = torch.nn.Linear(500,100)
        self.fc3    = torch.nn.Linear(100,100)
        self.fc_out = torch.nn.Linear(100,3)
        
    def forward(self, batch):
        x1   = F.relu(self.fc1(batch), True)
        code = F.relu(self.fc2(x1),    True)
        x2   = F.relu(self.fc3(code),  True)
        clf  = F.softmax(self.fc_out(x2),-1)
        
        return clf, code



batch_size = 64
n_iter = 1200*10
source_model = BlobModel()
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(source_model.parameters(), lr=0.001)

for i in tqdm.tqdm(range(n_iter), unit=" batches"):
    
    ind    = np.random.choice(len(source_traindata),size=batch_size,replace=False)
    xbatch = torch.tensor(source_traindata [ind].astype(np.float32))
    ybatch = torch.tensor(source_trainlabel[ind])
    
    optim.zero_grad()
    outputs, latent_code = source_model(xbatch)
    
    loss = criterion(outputs, ybatch)
    #loss = criterion(outputs, batch.y)
    loss.backward()
    optim.step()
    
    #tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

#source_model.fit(source_traindata,
#                 source_trainlabel_cat,
#                 batch_size=128,
#                 epochs=100,
#                 validation_data=(target_traindata, target_trainlabel_cat))
#%% Evaluate Model trained on source data
source_model.eval()
subset = 200
with torch.no_grad():
    preds_train, smodel_source_feat = source_model(
            torch.tensor(source_traindata.astype(np.float32)))
    
    smodel_source_feat = smodel_source_feat[:200]
    preds_train = torch.argmax(preds_train,dim=1)
    
    preds_targettrain, smodel_target_feat = source_model(
            torch.tensor(target_traindata.astype(np.float32)))
    smodel_target_feat = smodel_target_feat[:subset]
    preds_targettrain = torch.argmax(preds_targettrain,dim=1)
    
    source_acc = torch.mean((preds_train== torch.tensor(source_trainlabel)).type(torch.float))
    target_acc = torch.mean((preds_targettrain == torch.tensor(target_trainlabel)).type(torch.float))
    print("")
    print("source acc using source model", source_acc)
    print("target acc using source model", target_acc)
    
#%% deepjdot model and training
import DeepJDOT

batch_size=128
sloss = 2.0; tloss=1.0; int_lr=0.002; jdot_alpha=5.0
# DeepJDOT model initalization
al_model = DeepJDOT.Deepjdot(source_model, batch_size, n_class, optim=None,allign_loss=1.0,
                      sloss=sloss,tloss=tloss,int_lr=int_lr,jdot_alpha=jdot_alpha,
                      lr_decay=True,verbose=1)
# DeepJDOT model fit
losses, tacc = al_model.fit(source_traindata, source_trainlabel, target_traindata,
                            n_iter=1500,cal_bal=False)


#%% accuracy assesment
tarmodel_sacc = al_model.evaluate(source_traindata, 
                                  source_trainlabel)    
acc = al_model.evaluate(target_traindata, target_trainlabel)
print("source loss & acc using source+target model", tarmodel_sacc)
print("target loss & acc using source+target model", acc)


#%% intermediate layers of source and target domain for TSNE plot of target (DeepJDOT) model
source_model.eval()
subset = 200
with torch.no_grad():
    al_sourcedata = al_model.predict(source_traindata[:subset,])[1]
    al_targetdata = al_model.predict(target_traindata[:subset,])[1]

#%% function for TSNE plot (source and target are combined)
def tsne_plot(xs, xt, xs_label, xt_label, subset=True, title=None, pname=None):

    num_test=100
    if subset:
        combined_imgs = np.concatenate([xs[0:num_test], xt[0:num_test]])
        combined_labels = np.concatenate([xs_label[0:num_test],xt_label[0:num_test]])
        combined_labels = combined_labels.astype('int').T
    print(combined_labels.shape)
    print(combined_imgs.shape)
    
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(combined_imgs)
    plt.figure(figsize=(10, 10))
    plt.scatter(source_only_tsne[:num_test,0], source_only_tsne[:num_test,1],
                c=combined_labels[:num_test], s=75, marker='o', alpha=0.5, label='source train data')
    plt.scatter(source_only_tsne[num_test:,0], source_only_tsne[num_test:,1], 
                c=combined_labels[num_test:],s=50,marker='x',alpha=0.5,label='target train data')
    plt.legend(loc='best')
    plt.title(title)

#%% TSNE plots of source model and target model
title = 'tsne plot of source and target data with source model\n2D blobs visualization (shape=domain, color=class)'
tsne_plot(smodel_source_feat, smodel_target_feat, source_trainlabel,
          target_trainlabel, title=title)

title = 'tsne plot of source and target data with source+target model\n2D blobs visualization (shape=domain, color=class)'
tsne_plot(al_sourcedata, al_targetdata, source_trainlabel, 
          target_trainlabel, title=title)
   