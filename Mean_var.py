#-----------------------------------------------------------------------------
"""Mean-Variance 
Portfolio Class"""
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

"""1: 
Import Library"""
#----------------
from dx import *
#----------------

import systemicRiskMeasures as srm 
US_sectors= pd.load('USsectors')
US_sectors_returns= srm.logreturns(Returns=US_sectors)
      


"""2: 
Market Environment and Portfolio Object"""
#----------------
ma = market_environment('ma', dt.date(2010, 1, 1))
ma.add_list('symbols', ['AAPL', 'GOOG', 'MSFT', 'FB'])
ma.add_constant('source', 'google')
ma.add_constant('final date', dt.date(2014, 3, 1))

port = mean_variance_portfolio('am_tech_stocks', US_sectors)  # instantiates the portfolio class and retrieves all the time series data needed
port.get_weights()
#----------------

"""3: 
Set Weights"""
#----------------
port.set_weights([0.6, 0.2, 0.1, 0.1])
#----------------

"""4: 
Monte Carlo Simulation"""
#----------------
# Monte Carlo simulation of portfolio compositions
print '-------------------------------- 1  MONTE CARLO SIMULATION --------------------------------'

rets = []
vols = []

for w in range(500):
    weights = np.random.random(4)
    weights /= sum(weights)
    r, v, sr = port.test_weights(weights)
    rets.append(r)
    vols.append(v)

rets = np.array(rets)
vols = np.array(vols)

import matplotlib.pyplot as plt

plt.scatter(vols, rets, c=rets / vols, marker='o')
plt.grid(True)
plt.suptitle('Monte Carlo Simulation')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()
#--------------

"""5: 
Optimsise Portfolio relative to risk"""
#------------
print '-------------------------------- 2  Optimise Portfolio relative to risk --------------------------------'
port.optimize('Return', constraint=0.225, constraint_type='Exact') # interpretes volatility constraint as equality
print port


"""6: 
Optimsise Portfolio relative to return"""
print '--------------------------------3 Optimise Portfolio relative to return --------------------------------'
port.optimize('Vol', constraint=0.20, constraint_type='Bound')  # interpretes return constraint as inequality (upper bound)
print port

"""7: 
Optimise Shapre"""
#-------------
print '--------------------------------4 Optimise Shapre --------------------------------'
port.optimize('Sharpe')
print port
#------------


"""8: 
Efficient Frontier"""
#----------------
print '--------------------------------5 EFFICIENT FRONTIER --------------------------------'
evols, erets = port.get_efficient_frontier(100) # 100 points of the effient frontier
plt.scatter(vols, rets, c=rets / vols, marker='o')
plt.scatter(evols, erets, c=erets / evols, marker='x')
plt.grid(True)
plt.suptitle('Efficient Frontier')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()
#----------------

"""8: 
Capital Market Line"""
#----------------
print '--------------------------------6 CAPITAL MARKET LINE --------------------------------'
cml, optv, optr = port.get_capital_market_line(riskless_asset=0.05)
plt.figure(figsize=(8, 4))
plt.plot(evols, erets, lw=2.0, label='efficient frontier')
plt.plot((0, 0.4), (cml(0), cml(0.4)), lw=2.0, label='capital market line')
plt.plot(optv, optr, 'r*', markersize=10, label='optimal portfolio')
plt.legend(loc=0)
plt.grid(True)
plt.ylim(0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.show()

optr #Opt return

optv #opt vol
#----------------

"""9: 
Portfolio Composition"""
#----------------
print '--------------------------------7Portfolio Composition Optimisation --------------------------------'
print '---- Optimise Portfolio relative to optimised return'
port.optimize('Vol', constraint=optr, constraint_type='Exact')
print port

#or
print '---- Optimise Portfolio relative to optimised vol'
port.optimize('Return', constraint=optv, constraint_type='Exact')
print port
#----------------




















"""import sklearn.covariance 
import numpy as np
import pandas as pd

#1
FamaFrench49= pd.load('FenFrench49')   

#2 
avr_returns= np.array((FamaFrench49.mean()))

#3
sigma= sklearn.covariance.ledoit_wolf(FamaFrench49)
"""
