#import os
#import pandas as pd
import numpy as np
from scipy import stats
from scipy.io import loadmat
import matplotlib.pyplot as plt 
#from scipy.stats import multivariate_normal
from collections import Counter
data = loadmat('ex8data1.mat')

X = data['X']
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1])

Xval = data['Xval']
Yval = data['yval']

def gaussian_prop(X):
    mean = X.mean(axis=0)
    sig = X.var(axis=0)
    return mean,sig

def threshold(pval,yval):
    epsilon = 0
    best_f1 = 0
    step = (pval.max() - pval.min())/1000
    for e in np.arange(pval.min(),pval.max(),step):
        pred = pval < e
        
        tp = np.sum(np.logical_and(pred==1,yval==1)).astype(float)
        fp = np.sum(np.logical_and(pred==1,yval==0)).astype(float)
        fn = np.sum(np.logical_and(pred==0,yval==1)).astype(float)
        
        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        
        f1 = (2*prec*recall)/(prec+recall)
        
        if f1 > best_f1:
            best_f1 = f1
            epsilon = e
    return epsilon,best_f1

    
 
mu,sigma = gaussian_prop(X)

p = np.zeros((X.shape[0],X.shape[1]))
m = X.shape[1]
for i in range(m):
    p[:,i] = stats.norm(mu[i],sigma[i]).pdf(X[:,i])

#p[:,0] = stats.norm(mu[0],sigma[0]).pdf(X[:,0])
#p[:,1] = stats.norm(mu[1],sigma[1]).pdf(X[:,1])

pval = np.zeros((Xval.shape[0],X.shape[1]))
for i in range(m):
    pval[:,i] = stats.norm(mu[i],sigma[i]).pdf(Xval[:,i])
#pval[:,0] = stats.norm(mu[0],sigma[0]).pdf(Xval[:,0])
#pval[:,1] = stats.norm(mu[1],sigma[1]).pdf(Xval[:,1])

epsilon,f1 = threshold(pval,Yval)

outliners = np.where(p<epsilon)
dic = Counter(outliners[0])
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1])
ax.scatter(X[outliners[0],0],X[outliners[0],1],s=50,color='r',marker='x')