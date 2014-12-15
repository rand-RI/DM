#stage1: IMPORT LIBRARY
import pandas.io.data as pdio      #import pandas.io.data library
import systemicRiskMeasures as srm #import Systemic Risk Measures library
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt 
import numpy as np

#Stage2: Import Data

    #Mahalanobis Distance
Historical_Prices= pd.load('returns')
Adjusted_Close_Prices= Historical_Prices['Adj Close'].dropna()
returns= srm.logreturns(Adjusted_Close_Prices)

returns_Figure5= pd.load('returnsMD_Figure5')

    #Correlation Surprise


    
    #Absorption Ratio
FamaFrench49= pd.load('FenFrench49')
MSCIUS_PRICES= pd.load('MSCIUSA')


#Stage 2: Import Systemic Risk Measures and Run Pulled Data
SRM_mahalanobis= srm.MahalanobisDist(returns)#define Mahalanobis Distance Formula
SRM_correlationsurprise= srm.Correlation_Surprise(returns)#define Correlation Surprise Score
SRM_absorptionratio= srm.Absorption_Ratio(FamaFrench49)#define Absorption Ratio
systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio] # group systemic risk measures
srm.print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES)

#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
 #   fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
  #  fig.savefig("{}.jpg".format(sysRiskMeasure))


#what needs to be down
#mahalanobis Distance
#nneds to colour the top 75% percentile

srm.MahalanobisDist_Table1(returns)

        
#srm.MahalanobisDist_Table1(SRM_mahalanobis)['5 Day']
#A=srm.MahalanobisDist_Table1(SRM_mahalanobis)['5 Day'].max()

#B=srm.MahalanobisDist_Table1(SRM_mahalanobis)['5 Day'].min()
#a=0
#b=1

#normalised_data=[]
#for i in range(len(srm.MahalanobisDist_Table1(SRM_mahalanobis)['5 Day'])):
        
#    x=srm.MahalanobisDist_Table1(SRM_mahalanobis)['5 Day'][i]
#    normailse_value=(a+(x-A)*(b-a))/(B-A)
    
#    normalised_data.append(normailse_value)
    
#pd.DataFrame(normalised_data,srm.MahalanobisDist_Table1(SRM_mahalanobis)['5 Day'].index,columns=list('N'))
    

















