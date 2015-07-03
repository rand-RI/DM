 
""" This Python scripts is used to generate systemic risk returns from systemic risk modules
    imported from systemicRiskMeasures as srm  in or each Systemic Risk Paper
                                                                                """


"""STAGE 1: 
IMPORT LIBRARY"""
#-------------------------
import pandas as pd
import systemicRiskMeasures1 as srm   
import matplotlib.pyplot as plt    
                                      #Import Systemic Risk Measures library
#----------------------------------------------------------------------------------------------------------------------------------


"""STAGE 2: 
IMPORT DATA"""
#--------------------------
Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M')[155:456]
Rebalanced_portfolio= Balanced_port[101:]
w=10
window=w*12

"""Returns"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

FmmaFrench47_from1926=(((pd.load('FF49_1926').resample('M'))/100))
FmmaFrench47_from1926=FmmaFrench47_from1926[607+155: (len(FmmaFrench47_from1926)-7)] # Bring Forward to Same data
FmmaFrench47_from1926.index
noa=47

#Need to push in front by 41 months due to cut off
FmmaFrench47_from1926=FmmaFrench47_from1926[43:]

#OR INSERT THE LOG RETURNS
OPTIMALWEIGHTS=[]
MINVARWEIGHTS=[]
for i in range(window,len(FmmaFrench47_from1926)):
#for i in range(60,len(FmmaFrench47_from1926)):

    """Return Data""" 
    rets=FmmaFrench47_from1926[0:i]
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
    OPTIMALWEIGHTS.append(SHARPE_WEIGHTS)
        
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
    MINVARWEIGHTS.append(Minimum_variance_weights)
    
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
    print ['Weighting Iteration', i]
        


#-------------------------------------------------------    

FmmaFrench47_from1926=FmmaFrench47_from1926[window:] #adjust to start after five years for window
  
  #2 Optimal
FmmaFrench47_from1926_SHARPEW=(FmmaFrench47_from1926.values)*OPTIMALWEIGHTS
q=pd.DataFrame(FmmaFrench47_from1926_SHARPEW, index=FmmaFrench47_from1926.index)
q.columns=[list(FmmaFrench47_from1926.columns.values)]
SHARPE_Weight_Returns=q.sum(1)

    #3 Minimum Variance 
FmmaFrench47_from1926_SHARPEW=(FmmaFrench47_from1926.values)*MINVARWEIGHTS
p=pd.DataFrame(FmmaFrench47_from1926_SHARPEW, index=FmmaFrench47_from1926.index)
p.columns=[list(FmmaFrench47_from1926.columns.values)]
SHARPE_Weight_Returns=p.sum(1)





#FmmaFrench47_from1926_MINW=FmmaFrench47_from1926*MINVARWEIGHTS
#MINVARIANCE_Weight_Returns=FmmaFrench47_from1926_MINW.sum(1)
    

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
    #1.Equal_Weight 
FmmaFrench47_from1926_EqualW=FmmaFrench47_from1926*(1./47)
Equal_Weight_Returns=FmmaFrench47_from1926_EqualW.sum(1)

    

#Cumulative Returns
Net_return_values_EW=[]
for i in range(2,len(Equal_Weight_Returns)):
    Net_return=Equal_Weight_Returns[0:i].sum()
    Net_return_values_EW.append(Net_return)
Net_returns_EW=pd.DataFrame(Net_return_values_EW,index=Equal_Weight_Returns[2:].index) 
#
Net_return_values_SHARPE=[]
for i in range(2,len(SHARPE_Weight_Returns)):
    Net_return=SHARPE_Weight_Returns[0:i].sum()
    Net_return_values_SHARPE.append(Net_return)
Net_returns_SHARPE=pd.DataFrame(Net_return_values_SHARPE,index=SHARPE_Weight_Returns[2:].index) 
#
Net_return_values_MIN=[]
for i in range(2,len(MINVARIANCE_Weight_Returns)):
    Net_return=MINVARIANCE_Weight_Returns[0:i].sum()
    Net_return_values_MIN.append(Net_return)
Net_returns_MIN=pd.DataFrame(Net_return_values_MIN,index=MINVARIANCE_Weight_Returns[2:].index) 


SP500=Rebalanced_portfolio['^GSPC']
Net_return_values_RP=[]
for i in range(2,len(SP500)):
    Net_return=SP500[0:i].sum()
    Net_return_values_RP.append(Net_return)
Net_returns_RP=pd.DataFrame(Net_return_values_RP,index=SP500[2:].index)   

fig= plt.figure(1, figsize=(10,3))
plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
plt.xlabel('Year')                                                          #label x axis Year
plt.ylabel('Cumulative Returns')                                                         #label y axis Index
plt.suptitle('Comparision of Returns',fontsize=12)   
plt.plot(Net_returns_EW.index,Net_returns_EW.values, label='Equal Weight', linestyle='--')
plt.plot(Net_returns_SHARPE.index,Net_returns_SHARPE.values, label='Optimal', linestyle='--')
plt.plot(Net_returns_MIN.index,Net_returns_MIN.values, label='MIN VAR', linestyle='--')
plt.plot(Net_returns_RP.index[:190],Net_returns_RP.values[:190], label='S&P 500')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
plt.grid()
plt.show()



#-------------------------
#----------------------------------------------------------------------------------------------------------------------------------

"""
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
plt.show()
"""





