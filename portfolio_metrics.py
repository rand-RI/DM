# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 17:32:58 2014

@author: xxklow

Portfolio metrics is a module for evaluating portfolio performance.

"""

def sharpe(ret,cash=np.array([[0]])):
    """
    Calculates the sharpe ratio
    """                      
    return (ret.mean()-cash.mean())/ret.std()
                          
def VaR(ret,beta=0.05):
    """
    Calculates the Value-At-Risk
    """
    return  abs(ret.quantile(beta))

def CVaR(ret,beta=0.05):
    """
    Calculates the Conditional Value-At-Risk or Expected Shortfall
    """
    mask = ret<=VaR(ret,beta)
    return abs(ret[mask].mean())

def omega_k(ret,MAR=0):
    """
    Calculates the Omega Ratio using the Kappa formula
    """
    return kappa(ret,MAR,1)+1
                
def sortino_k(ret,MAR=0):
    """
    Calculates the Sortino Ratio using the Kappa formula
    """
    return kappa(ret,MAR,2)                

def omega(ret,MAR=0):
    """
    Calculates the Omega Ratio.  Captures the idea that investors want more than less.
    """
    return lpm(-ret,-MAR,1)/lpm(ret,MAR,1)
                
def sortino(ret,MAR=0):
    """
    Calculates the Sortino Ratio.  Risk-adjusted returns focusing on downside std. deviation
    """
    return (ret.mean()-MAR)/root(lpm(ret,MAR,2),2)
    
def upsidePot(ret,MAR=0):
    """
    Calculates upside potential of asset returns
    """
    return lpm(-ret,-MAR,1)/root(lpm(ret,MAR,2),2)
                    
                
def kappa(ret,MAR,order):
    """
    Generalized downside risk-adjusted performance measure
    The indicator can become any risk-adjusted return through the 'Order' parameter
    Omega measure : order=1 and +1 to the result
    Sortino Ratio : order=2
    """
    lpmVal = lpm(ret,MAR,order)
    k = (ret.mean()-MAR)/root(lpmVal,order)           
    return k 

def lpm(ret,MAR,order):
    """
    Calculates lower-partial moments    
    """
    lpmVal =  ((ret<MAR)*(MAR-ret)**order).mean()    
    return lpmVal
    
def root(n,order=1):
    """
    Calculates the root of any value given
    """
    if order == 0:
        return 1
    elif order == 1:
        return n
    else:
        return abs(np.roots([1]+[0]*(order-1)+[-n]))[1]
        
