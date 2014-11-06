#stage1: IMPORT LIBRARY
import pandas.io.data as pdio      #import pandas.io.data library
import systemicRiskMeasures as srm #import Systemic Risk Measures library
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt 
import numpy as np
#stage2: IMPORT DATA WITH ANY NUMBER OF PORTFOLIOS 

          #Must arrange tickers in alphabetical order
symbols = ['^AORD','^ATX','^BFX','^BSESN','^BVSP','^FCHI','^GDAXI','^GSPC','^GSPTSE','^HSI','^JKSE','^KLSE','^KS11','^MERV','^MXX','^N225','^SSEC','^STI','^TWII'] # List all stock symbols to download in alphabetical order


#stage3: DOWNLOAD DATA AND CALCULATE RETURN VALUES
Start_Date='11/1/1980'#MM,DD,YY
End_Date='12/15/2009'#MM,DD,YY
Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date) # Download data from YAHOO as a pandas Panel object
Adjusted_Close_Prices = Historical_Prices['Adj Close'].dropna()  # Scrape adjusted closing prices as pandas DataFrane object while also removing all Nan data

#daily returns:
returns = np.log(Adjusted_Close_Prices/Adjusted_Close_Prices.shift(1)).dropna()  # Continuously compounded returns while also removing top row of Nan data
#monthly returns


#stage4: Import Systemic Risk Measures
SRM_mahalanobis= srm.MahalanobisDist(returns)[0]
      #define Mahalanobis Distance Formula
SRM_correlationsurprise= srm.Correlation_Surprise(returns)#define Correlation Surprise Score
SRM_absorptionratio= srm.Absorption_Ratio(returns)#define Absorption Ratio

systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio] # group systemic risk measures

srm.print_systemic_Risk(systemicRiskMeasure)



#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
 #   fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
  #  fig.savefig("{}.jpg".format(sysRiskMeasure))