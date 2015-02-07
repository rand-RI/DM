"""pg 306 Python for Finance: Big Data"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd

"""Step1:
Import Data as DataFrame"""
#-------------
FamaFrench49= pd.load('FenFrench49')                # Import DataFrame
data=FamaFrench49
rets=data


noa=len(data.columns)                                #Generate Len of Columns
#log returns
    #rets= np.log(data/data.shift(1))
#-------------

"""Step2:
Portfolio Optim"""
import scipy.optimize as sco
import sklearn.covariance
#Step1:
#----------------
#Function returns major portfolio statistics for an input weights vector/array
def statistics(weights):               
    """ Returns portfolio statistics.
    Parameters
    ==========
    weights : array-like
    weights for different securities in portfolio
    Returns
    =======
    pret : float
    expected portfolio return
    pvol : float
    expected portfolio volatility
    pret / pvol : float
    Sharpe ratio for rf=0
    """
    weights = np.array(weights)
    mean=rets.mean()
    covariance=sklearn.covariance.ledoit_wolf(rets)[0]
    pret = np.sum(mean * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(covariance * 252, weights)))
    return np.array([pret, pvol, pret / pvol])
    
#Minimise Variance
def min_func_variance(weights):
    return statistics(weights)[1] ** 2
#----------------

#Step2: Add Constraints
#-------------------
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #list constraint that all weights add to 1    
bnds = tuple((0, 1) for x in range(noa)) #bound weights so that values are only within 0 and 1
Equal_weights= noa * [1. / noa,]  #Input an equal distribution of weights
#-------------------

#Step3: Generate Optimised Returns
#-------------------
optv = sco.minimize(min_func_variance, Equal_weights, method='SLSQP', bounds=bnds,  constraints=cons)
print 'Optimised Vol Weights'
print optv['x'].round(3)
print #
print 'Expected Returns:' ,statistics(optv['x']).round(3)[0]
print 'Volatility:' ,statistics(optv['x']).round(3)[1]
print 'Sharpe:' ,statistics(optv['x']).round(3)[2]





