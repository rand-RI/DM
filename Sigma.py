"""Sigma"""
import pandas as pd
import numpy as np

FamaFrench49= pd.load('FenFrench49')   
x=FamaFrench49.values




t= x.shape[0]
n= x.shape[1]
meanx= x.mean(0)
e_size= np.zeros(shape=(1,x.shape[0]))
e_filled= e_size.fill(1)
e= np.transpose(e_size)
x=x-meanx*e
xmkt= np.transpose(np.transpose(x).mean(0))
xmkt=np.transpose((xmkt*e[1:2]))
    
sample= np.divide((np.cov(np.transpose((np.concatenate((x, xmkt), 1)))))*(t-1),t) 
covmkt= (sample[:,n][0:n])
covmkt=np.reshape(covmkt, (n, 1))
varmkt= sample[:,n][n:n+1]
sample= sample[:,0:n]
sample= sample[0:n,:]
prior=(covmkt*np.transpose(covmkt))/varmkt
np.fill_diagonal(prior, np.diagonal(sample))

#shrinkage
m_size= np.zeros(shape=(1,n))
m_filled= m_size.fill(1)
matrix_ones= m_size

c= np.square(np.linalg.norm(sample-prior, ord= 'fro'))
y= np.square(x)
p= (1./t)*np.dot(np.transpose(y),y).sum(0).sum() - np.square(sample).sum(0).sum()
rdiag=(1./t)*np.square(y).sum(0).sum() - np.square(np.diagonal(sample)).sum()
z= x*xmkt
v1= (1./t)*np.dot(np.transpose(y),z) - np.dot(covmkt,matrix_ones)*sample                #matrix ones is 1,49 matrix of ones
roff1= ((v1*np.transpose(np.dot(covmkt,matrix_ones))).sum(0).sum())/varmkt - ((np.reshape(v1.diagonal(),(n,1))*covmkt).sum())/varmkt
v3=(1./t)*np.dot(np.transpose(z),z) - varmkt*sample
roff3=((v3*(covmkt*np.transpose(covmkt))).sum(0).sum())/np.square(varmkt) -((np.reshape(v3.diagonal(),(n,1))*np.square(covmkt)).sum())/np.square(varmkt)
roff= 2*roff1 - roff3
r= rdiag+roff
k=(p-r)/c
shrinkage= np.reshape(np.max(np.reshape(np.min(k/t),(1,1))),(1,1))

sigma= shrinkage*prior + (1-shrinkage)*sample

print shrinkage





   
