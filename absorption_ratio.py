from pylab import *
from pandas.io.data import *
from pandas import *
from pandas.tools.plotting import *
import numpy as np



#stage1: IMPORT DATA WITH ANY NUMBER OF PORTFOLIOS
symbols = ['^AORD','^HSI','^N225'] # List all stock symbols to download in alphabetical order
stock_data = get_data_yahoo(symbols,start='1/1/2005',end='1/1/2014') # Download data from YAHOO as a pandas Panel object

adj_close  = stock_data['Adj Close']         # Scrape adjusted closing prices as pandas DataFrane object
returns = log(adj_close/adj_close.shift(1))  # Continuously compounded returns

#stage2: subtract the mean
means= returns.mean()                       #Calculate return means
diff_means= returns.subtract(means)             

#stage3: CALCULATE COVARIANCE MATRIX
return_covariance= diff_means.cov()             #Generate Covariance Matrix

#stage4: CALCULATE EIGENVECTORS AND EIGENVALUES
ev_values,ev_vector= linalg.eig(return_covariance)         #generate eigenvalues and vectors 
ev_values_low_to_high=np.sort(ev_valuess)       #sort lowest to highest
ev_values_high_to_low= ev_values_low_to_high[::-1]          #sort highest to lowest




  
