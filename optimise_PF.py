#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from matplotlib.finance import quotes_historical_yahoo
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import fmin

#Step2: define a few functions
#function 1:

#function 2: estimate portfolio variance
def portfolio_var(R,W):
    cor= sp.corrcoef(R.T)
    std_dev=sp.std(R,axis=0)
    var=0.0
    for i in xrange(n):
        for j in xrange(n):
            var += W[i]*W[j]*std_dev[i]*std_dev[j]*cor[i,j]
    return var
#function 3: estimate Sharpe ratio
def sharpe(R,W):
    var= portfolio_var(R,W)
    mean_return=np.mean(R,axis=0)
    ret=sp.array(mean_return)
    return(sp.dot(W,ret)-rf)/np.sqrt(var)
#function 4: for given n-1 weights, return a negative sharpe ratio
def negative_sharpe_n_minus_1_stock(W):
    W2=sp.append(W,1-sum(W))
    return -sharpe(R,W2)
        
#Step 3: generate a return matrix(annul return)

#import DataFrame
FamaFrench49= pd.load('FenFrench49')
rf= 0.0000  #annual risk-free rate
x2=FamaFrench49
n=len(x2.columns)
ticker= x2.columns

    
R=sp.array(x2)
print('Efficient portfolio(mean-variance): ticker used')
print (ticker.values)
print #------------
print ('Sharpe given equal weights')
equal_w=sp.ones(n, dtype=float)*1.0/n
print(sharpe(R,equal_w))
print #------------
#for n stocks, we could only choose n-1 weights
wO=sp.ones(n-1,dtype=float)*1.0/n
w1=fmin(negative_sharpe_n_minus_1_stock,wO)
final_W=sp.append(w1,1-sum(w1))
final_sharpe=sharpe(R,final_W)
print ('Optimal weights are ')
print (final_W)
print #------------
print ('final Sharpe Ratio is ')
print (final_sharpe)
        