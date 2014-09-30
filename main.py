#stage1: IMPORT LIBRARY
from pylab import *
from pandas.io.data import *
from pandas import *
from mahalanobis_distance import MahalanobisDist   #import Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
from correlation_surprise import Correlation_Surprise  # import Kinlaw and Turkington - 2012 - Correlation Surprise

#stage2: IMPORT DATA WITH ANY NUMBER OF PORTFOLIOS

symbols = ['^AORD','^HSI','^N225','^NZ50','^FTSE'] # List all stock symbols to download in alphabetical order
stock_data = get_data_yahoo(symbols,start='1/1/2004',end='1/1/2014') # Download data from YAHOO as a pandas Panel object

#stage3: Import Systemic Risk Measures
MahalanobisDist(stock_data)
Correlation_Surprise(stock_data)





