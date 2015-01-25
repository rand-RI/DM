import pandas as pd
import numpy as np
import math as mth

FamaFrench49= pd.load('FenFrench49')      

#1
Points=FamaFrench49= pd.load('FenFrench49')   

#2
Center_Points= Points.subtract(Points.mean())

#3
Covariance= Center_Points.cov()

#4
ev_values,ev_vector = np.linalg.eig(Covariance)

#5
ev_values_sort_high_to_low = ev_values.argsort()[::-1]                         
ev_values_sort=ev_values[ev_values_sort_high_to_low]                   #sort eigenvalues from highest to lowest
ev_vectors_sorted= ev_vector[:,ev_values_sort_high_to_low]

#6
from sklearn.decomposition import PCA
pca = PCA(n_components=3, whiten=False).fit(Center_Points)

#7
Eigenvectors= pca.components_
Eigenvalues= pca.explained_variance_ratio_/np.linalg.norm(pca.explained_variance_ratio_)






