

FamaFrench49= pd.load('FenFrench49')     
sample=FamaFrench49

import numpy as np
        
u_all= sample.mean()
returns=sample

u_s_all=[]
for i in range(len(u)):
        
    u= u_all[i]       
    #Calculate u_m (shrinkage target)        
    e_size= np.zeros(shape=(1,len(sample.columns)))
    e_filled= e_size.fill(1)
    e= np.transpose(e_size)
    
    Covariance_Matrix_inverse= np.linalg.inv(returns.cov())
    
    
    u_m_numerator= np.dot((np.transpose(e)*Covariance_Matrix_inverse),u)
    u_m_denominator= np.dot((np.transpose(e)*Covariance_Matrix_inverse),e)
    u_m=np.divide(u_m_numerator, u_m_denominator)[0][0]                             #grab value out from array so it is a number
    
    
    n=len(sample.columns)
    T=len(returns)
    w_numerator= n+2
    w_denominator=n+2+T*(np.dot(np.transpose(u-u_m*e),(np.dot(Covariance_Matrix_inverse,u-u_m*e))))
    w=np.divide(w_numerator, w_denominator)[0][0]
    
    u_s=w*u_m +(1-w)*u
    u_s_all.append(u_s)