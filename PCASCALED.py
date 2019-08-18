# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:01:17 2019

@author: Ashwin Surya
"""

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

#'loading the data from MATLAB'
data=scipy.io.loadmat ('onehotenc.mat')
inputs=data['input'].T

out=np.random.randint(0, 1, size=(150012, 1))
out[0:10001]=1
out[10002:20003]=2
out[20003:30004]=3
out[30004:40005]=4
out[40005:50006]=5
out[50006:60007]=6
out[60007:70008]=7
out[70008:80009]=8
out[80009:90010]=9
out[90010:100011]=10
out[100011:]=11
df = pd.DataFrame(np.concatenate((inputs,out), 1))
df.columns=['Va','Vb','Vc','Ia','Ib','Ic','tar']

X=df[['Va','Vb','Vc','Ia','Ib','Ic']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
s_X=scaler.transform(X)

#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(s_X)
x_pca=pca.transform(s_X)
x_pca.shape
plt.figure(figsize=(8,8))
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['tar'],cmap='viridis')
plt.xlabel("First PCA comp")
plt.ylabel("Second PCA comp")
pca.components_

from tsne import bh_sne
X_2d = bh_sne(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['tar'])
