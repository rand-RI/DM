 
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
Start='19950131'
Start_Recession_Values='19950201'
End='20140131' #20140630 latest St Louis Recession data date


Recession_Values= pd.load('USARECM')    
Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M',how='sum').loc[Start:End]
Recession_Values= Recession_Values[Start_Recession_Values:] 

w=5
window=w*12
window_range= window
"""Returns"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

FmmaFrench47_from1926=(pd.load('FF49_1926').resample('M',how='sum')).loc[Start:End]
noa=47
length=len(FmmaFrench47_from1926)

#OR INSERT THE LOG RETURNS
OPTIMALWEIGHTS=[]
MINVARWEIGHTS=[]
for i in range(window,length):
#for i in range(60,len(FmmaFrench47_from1926)):

    """Return Data""" 
    rets=FmmaFrench47_from1926[0:i]
    #rets = np.log(data / data.shift(1))
    rets.mean() * 12
    rets.cov() * 12
    
    """Python Implementations of MV""" 
    weights = np.random.random(noa)
    weights /= np.sum(weights) # getting random weights and normalization
    weights
    np.sum( rets.mean() * weights ) * 12
      # expected portfolio return
    np.dot(weights.T, np.dot(rets.cov() * 12, weights))
    np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights)))
      # expected portfolio standard deviation/volatility
    
    """Simulate MV Combinations""" 
#    prets = []
#    pvols = []
#    for p in range (100):
#        weights = np.random.random(noa)
#        weights /= np.sum(weights)
#        prets.append(np.sum(rets.mean() * weights) * 252)
#        pvols.append(np.sqrt(np.dot(weights.T, 
#                            np.dot(rets.cov() * 252, weights))))
#    prets = np.array(prets)
#    pvols = np.array(pvols)
    
    
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
        pret = np.sum(rets.mean() * weights) * 12
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights)))
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
    
#    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#    bnds = tuple((0, 1) for x in weights)
#    def min_func_port(weights):
#        return statistics(weights)[1]
#    trets = np.linspace(0.0, 0.25, 50)
#    tvols = []
#    for tret in trets:
#        cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
#                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#        res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP', 
#                           bounds=bnds, constraints=cons)
#        tvols.append(res['fun'])
#    tvols = np.array(tvols)  
    print ['Weighting Iteration', i]
        


#-------------------------------------------------------    

FmmaFrench47_from1926=FmmaFrench47_from1926[window:] #adjust to start after five years for window
x=length-60
  #2 Optimal
FmmaFrench47_from1926_SHARPEW=(FmmaFrench47_from1926[0:x].values)*OPTIMALWEIGHTS
q=pd.DataFrame(FmmaFrench47_from1926_SHARPEW, index=FmmaFrench47_from1926[0:x].index)
q.columns=[list(FmmaFrench47_from1926.columns.values)]
SHARPE_Weight_Returns=q.sum(1)

    #3 Minimum Variance 
FmmaFrench47_from1926_SHARPMW=(FmmaFrench47_from1926[0:x].values)*MINVARWEIGHTS
p=pd.DataFrame(FmmaFrench47_from1926_SHARPMW, index=FmmaFrench47_from1926[0:x].index)
p.columns=[list(FmmaFrench47_from1926.columns.values)]
MIN_Weight_Returns=p.sum(1)

    

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

SHARPE_Weight_Returns=SHARPE_Weight_Returns/100+1
Net_return_values_R=[]
Initial_Amount=100
for i in range(1,len(SHARPE_Weight_Returns)+1):
    Net_return=(SHARPE_Weight_Returns[i-1:i]*Initial_Amount)[0]
    Net_return_values_R.append(Net_return)
    Initial_Amount=Net_return
Net_returns_R=pd.DataFrame(Net_return_values_R,index=SHARPE_Weight_Returns.index) 
Net_returns_R.save('MV') 

MIN_Weight_Returns=MIN_Weight_Returns/100+1
Net_return_values_R=[]
Initial_Amount=100
for i in range(1,len(MIN_Weight_Returns)+1):
    Net_return=(MIN_Weight_Returns[i-1:i]*Initial_Amount)[0]
    Net_return_values_R.append(Net_return)
    Initial_Amount=Net_return
Net_returns_M=pd.DataFrame(Net_return_values_R,index=MIN_Weight_Returns.index)     
Net_returns_M.save('MinV') 

fig= plt.figure(1, figsize=(10,3))
plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
plt.xlabel('Year')                                                          #label x axis Year
plt.ylabel('Price')  


plt.plot(Net_returns_R.index,Net_returns_R.values, label='MV', linestyle='--')
plt.plot(Net_returns_M.index,Net_returns_M.values, label='MinV')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
plt.grid()
plt.show()

cost=0.05 
#################
OP=np.array(OPTIMALWEIGHTS)
#OA=(OP[0:1])*0 + 1./47
#OP=np.append(OA,OP,axis=0)    
turnover=[]
for i in range(len(OP)-1):
    Turnover=OP[i+1:i+2]-OP[i:i+1]
    Abs_Turnover=np.absolute(Turnover).sum()
    turnover.append(Abs_Turnover)

SHARPE_Weight_Returns=SHARPE_Weight_Returns
Net_return_values_R=[]
Initial_Amount=100
#cost=0.005
for i in range(1,len(SHARPE_Weight_Returns)):
#+1):
    Net_return=(SHARPE_Weight_Returns[i-1:i]*Initial_Amount)[0]
    t=cost*turnover[i-1]*Net_return
    Net_return=Net_return-t 
    Net_return_values_R.append(Net_return)
    Initial_Amount=Net_return
Net_returns_RT=pd.DataFrame(Net_return_values_R,index=SHARPE_Weight_Returns[:len(SHARPE_Weight_Returns)-1].index) 
Net_returns_RT.save('MV_T') 

MIN_Weight_Returns=MIN_Weight_Returns
Net_return_values_R=[]
Initial_Amount=100
#cost=0.005
for i in range(1,len(MIN_Weight_Returns)):
#+1):
    Net_return=(MIN_Weight_Returns[i-1:i]*Initial_Amount)[0]
    t=cost*turnover[i-1]*Net_return
    Net_return=Net_return-t 
    Net_return_values_R.append(Net_return)
    Initial_Amount=Net_return
Net_returns_RTMV=pd.DataFrame(Net_return_values_R,index=MIN_Weight_Returns[:len(SHARPE_Weight_Returns)-1].index) 
Net_returns_RTMV.save('MinV_T') 

fig= plt.figure(1, figsize=(10,3))
plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
plt.xlabel('Year')                                                          #label x axis Year
plt.ylabel('Price')  

plt.plot(Net_returns_RT.index,Net_returns_RT.values, label='MVT', linestyle='--')
plt.plot(Net_returns_RTMV.index,Net_returns_RTMV.values, label='MinVT', linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
plt.grid()
plt.show()
 
 
 
 
 
 
 
 
 

#Cumulative Returns
"""
Net_return_values_SHARPE=[]
for i in range(2,len(SHARPE_Weight_Returns)):
    Net_return=SHARPE_Weight_Returns[0:i].sum()
    Net_return_values_SHARPE.append(Net_return)
Net_returns_SHARPE=pd.DataFrame(Net_return_values_SHARPE,index=SHARPE_Weight_Returns[2:].index) 
#
Net_return_values_MIN=[]
for i in range(2,len(MIN_Weight_Returns)):
    Net_return=MIN_Weight_Returns[0:i].sum()
    Net_return_values_MIN.append(Net_return)
Net_returns_MIN=pd.DataFrame(Net_return_values_MIN,index=MIN_Weight_Returns[2:].index) 
"""
#SP500=Balanced_port['^GSPC']
#Net_return_values_RP=[]
#for i in range(2,len(SP500)):
#    Net_return=SP500[0:i].sum()
#    Net_return_values_RP.append(Net_return)
#Net_returns_RP=pd.DataFrame(Net_return_values_RP,index=SP500[2:].index)   
#Start='20000131'
#End='20140131'
#Net_returns_RP=Net_returns_RP.loc[Start:End]


                                                   #label y axis Index
 
#plt.plot(Net_returns_SP.index,Net_returns_SP.values, label='Switch Strad', linestyle='--')
#plt.plot(Net_returns_R.index,Net_returns_R.values, label='S&P 500')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
#plt.grid()
#plt.show()
#print ['Over Index of', Net_returns_R.index]







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





