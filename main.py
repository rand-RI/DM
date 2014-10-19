#stage1: IMPORT LIBRARY
import pandas.io.data as pdio      #import pandas.io.data library
import systemicRiskMeasures as srm #import Systemic Risk Measures library
import matplotlib.pyplot as plt
#stage2: IMPORT DATA WITH ANY NUMBER OF PORTFOLIOS 

          #Must arrange tickers in alphabetical order
symbols = ['EZA','FTSEMIB.MI','RTS.RS','^AORD','^ATX','^BFX','^BSESN','^BVSP','^FCHI','^FTSE','^GDAXI','^GSPTSE','^IXIC','^JKSE','^KS11','^MERV','^MXX','^N225','^SSEC'] # List all stock symbols to download in alphabetical order
#G20 MEMBERS#
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
#South Africa: EZA
#Turkey:
#United Kingdom: ^FTSE
#United States of America:'^IXIC'
#the European Union: 
#^ATX
#^BFX
#^FCHI
#^GDAXI
#
#
#^OMXSPI
#^SSMI
#
#
#
#

#stage3: DOWNLOAD DATA AND CALCULATE RETURN VALUES

Historical_Prices = pdio.get_data_yahoo(symbols,start='1/1/1980',end='1/10/2009') # Download data from YAHOO as a pandas Panel object
Adjusted_Close_Prices = Historical_Prices['Adj Close'].dropna()  # Scrape adjusted closing prices as pandas DataFrane object while also removing all Nan data
returns = np.log(Adjusted_Close_Prices/Adjusted_Close_Prices.shift(1)).dropna()  # Continuously compounded returns while also removing top row of Nan data


#stage4: Import Systemic Risk Measures
SRM_mahalanobis= srm.MahalanobisDist(returns)       #define Mahalanobis Distance Formula
SRM_correlationsurprise= srm.Correlation_Surprise(returns)#define Correlation Surprise Score
SRM_absorptionratio= srm.Absorption_Ratio(returns)#define Absorption Ratio

systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio]

#not sure what the below code does or how to set it up?
#for sysRiskMeasure in systemicRiskMeasure:
   # fig= print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
    # fig.savefig("{}.jpg".format(sysRiskMeasure))
    
    
#Plot SRM_mahalanobis
plt.xticks(rotation=50)
plt.xlabel('Year')
plt.ylabel('Index')
plt.suptitle('Historical Turbulence Index Calculated from Daily Retuns of G20 Countries')
plt.bar(SRM_mahalanobis.index,SRM_mahalanobis.values, width=2)
plt.show()

#Plot Correlation_surprise
Correlation_Surprise= SRM_correlationsurprise[0]
Turbulence_Score= SRM_correlationsurprise[1]
plt.xlabel('Magnitude Surprise')
plt.ylabel('CorrelationSurprise')
plt.suptitle('Daily correlation surprise versus magnitude surprise')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.scatter(Turbulence_Score,Correlation_Surprise)      #Plot SRM_correlationsurprise

#not sure how to generate daily data for Absorption Ratio


#systemicRiskMeasure = [SRM_mahalanobis, SRM_correlationsurprise] # group systemic risk measures

#what I need to do is finish AB
#plot graphs using the SRM command from the github references
#remove TB from Correlation Surprise and just use MD

