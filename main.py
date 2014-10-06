#stage1: IMPORT LIBRARY
import pandas.io.data as pdio
import matplotlib.pyplot as plt
from mahalanobis_distance import MahalanobisDist   #import Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
from correlation_surprise import Correlation_Surprise  # import Kinlaw and Turkington - 2012 - Correlation Surprise


#stage2: IMPORT DATA WITH ANY NUMBER OF PORTFOLIOS

symbols = ['FTSEMIB.MI','^IXIC','RTS.RS','^AORD','^BSESN','^BVSP','^FCHI','^FTSE','^GDAXI','^GSPTSE','^JKSE','^KS11','^MERV','^MXX','^N225','^SSEC'] # List all stock symbols to download in alphabetical order
#G20 MEMBERS
#Argentina:^MERV is considered the most important index of Argentina's primary stock exchange "Buenos Aires Stock Exchange"
#Austrlaia: ^AORD is considered the oldest index of shares in Australia
#Brazil: ^BVSP is an index of 50 stocks that are traded on the Sao Paulo Stock Exchange in Brazil
#Canada:^GSPTSE is an index of the stock (equity) prices of the largest companies on the Toronto Stock Exchange in Canada
#China: ^SSEC represents all stocks that are traded on the Shanghai Stock Exchange
#France: ^FCHI is considered a benchmark French Stock Market index
#Germany: ^GDAXI
#India:^BSESN
#Indonesia:^JKSE
#Italy:FTSEMIB.MI
#Japan:^N225
#Republic of Korea: ^KS11
#Mexico:^MXX
#Russia: RTS.RS
#Saudi Arabia: ...
#South Africa:...
#Turkey:...
#United Kingdom: ^FTSE
#United States of America:'^IXIC'
#the European Union: need to collect all countries?

stock_data = pdio.get_data_yahoo(symbols,start='1/1/2000',end='1/10/2014') # Download data from YAHOO as a pandas Panel object

# Convert data frame prices into returns data
adj_close = stock_data['Adj Close'] # Scrape adjusted closing prices as pandas DataFrane object
returns = np.log(adj_close/adj_close.shift(1)) # Continuously compounded returns

# Calculate systemic risk measures
SRM_mahalanobis = srm.mahalanobis_distance(returns)
SRM_correlationsurprise = srm.correlationsurprise(returns)
SRM_absorptionratio = srm.absorptionratio(returns)

systemicRiskMeasure = [SRM_mahalanobis SRM_correlationsurprise SRM_absorptionratio]   # group systemic risk measures

for sysRiskMeasure in systemicRiskMeasure
	fig = print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
	fig.savefig("{}.jpg".format(sysRiskMeasure))

# Plotting the systemic risk measures

def print_systemicRiskMeasure:
	SRM_mahalanobis.plot()               #Plot MD
	plt.ylabel('Index')
	plt.xlabel('Year')






