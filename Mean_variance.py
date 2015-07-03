import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

FmmaFrench47_from1926=((pd.load('FF49_1926').resample('M'))[607+258: 607+448])/100
noa=47

### IF Stock DATA WANTED
"""
symbols = ['AAPL', 'GOOG', 'MSFT', 'DB', 'GLD']
  # the symbols for the portfolio optimization
noa = len(symbols)
import pandas.io.data as web
data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='google')['Close']
data.columns = symbols

(data / data.ix[0] * 100).plot(figsize=(8, 5))
"""



#OR INSERT THE LOG RETURNS

#------------------------------------------------------------------------------
"""Return Data""" 
rets=FmmaFrench47_from1926
#rets = np.log(data / data.shift(1))
rets.mean() * 252
rets.cov() * 252

"""Python Implementations of MV""" 
weights = np.random.random(noa)
weights /= np.sum(weights) # getting random weights and normalization
weights
np.sum( rets.mean() * weights ) * 252
  # expected portfolio return
np.dot(weights.T, np.dot(rets.cov() * 252, weights))
np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
  # expected portfolio standard deviation/volatility

"""Simulate MV Combinations""" 
prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T, 
                        np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)


"""Derive Efficient Frontier""" 

def statistics(weights):
    """
    Returns portfolio statistics.
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
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])
 
import scipy.optimize as sco
###SHAPRE
def min_func_sharpe(weights):
    return -statistics(weights)[2]
    
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))   
def min_portfolio_vola(weights):
    return statistics(weights)[1]
opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',bounds=bnds, constraints=cons)
SHARPE_WEIGHTS= opts['x'].round(3)
SHARPE_STATS=statistics(opts['x']).round(3)

###Min Variance port weights 
def min_func_variance(weights):
    return statistics(weights)[1] ** 2
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
noa * [1. / noa,]
optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
Minimum_variance_weights= optv['x'].round(3)
#Statistics: Expected return, vol and Sharpe
Minimum_variance_weights_STATS= statistics(optv['x']).round(3)


cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in weights)
def min_func_port(weights):
    return statistics(weights)[1]
trets = np.linspace(0.0, 0.25, 50)
tvols = []
for tret in trets:
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP', 
                       bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)    
    
plt.figure(figsize=(8, 4))
plt.xlim(0, 0.1)
plt.scatter(tvols, trets,c=trets / tvols, marker='x')
# efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],'r*', markersize=15.0)
# portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],'y*', markersize=15.0)
# minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')    


#-------------------------------------------------------    
