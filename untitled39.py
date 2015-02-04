import numpy as np
import systemicRiskMeasures as srm
import pandas as pd   

US_sectors= pd.load('USsectors')
US_sectors_returns= srm.logreturns(Returns=US_sectors)
      

#(1) absolute weights (MVO_abs_wtgs) 

x_t= [0.2,0.3,0.4]
N= int(np.shape(x_t)[0])
one_n= np.ones((1,N))
MVO_abs_wtgs= x_t/(np.absolute(np.transpose(one_n)*x_t))


## calcalcuate x_T
y=0.2
sigma= US_sectors_returns.cov()
sigma_inv=np.linalg.inv(sigma)
u=np.reshape(np.array(US_sectors_returns.mean()), (1,US_sectors_returns.shape[1]))

x_t=(1/y)*sigma_inv*u




#(2) relative weights (MVO_rel_wtgs)u= returns.mean()
u_t=np.reshape(np.array(US_sectors_returns.mean()), (US_sectors_returns.shape[1],1))
sigma= US_sectors_returns.cov()
one_n= np.ones((1,US_sectors_returns.shape[1]))
MVO_rel_wtgs= (np.linalg.inv(sigma)*u_t)/(one_n*np.linalg.inv(sigma)*u_t)



