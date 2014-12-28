                             #1: IMPORT LIBRARY
import pandas.io.data as pdio      
import systemicRiskMeasures as srm #import Systemic Risk Measures library
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt 
import numpy as np

                            #2: IMPORT DATA

    #Mahalanobis Distance
#Data
Historical_Prices= pd.load('returns')
Adjusted_Close_Prices= Historical_Prices['Adj Close'].dropna()
returns= srm.logreturns(Adjusted_Close_Prices) #Daily returns of 

returns_Figure5= pd.load('returnsMD_Figure5') # Daily returns of World Equities, US small-capitalisation premium(Small-Large), growth Premium(Growth-Value) and HFRI Fund of Funds composite Index

    #Correlation Surprise
returns_Figure5= pd.load('returnsMD_Figure5') #need to gather data for this one(Datastream) #US equities MSCI secotors
returns_Figure5= pd.load('returnsMD_Figure5')#European Equities

    
    #Absorption Ratio
FamaFrench49= pd.load('FenFrench49') #51 sectors
MSCIUS_PRICES= pd.load('MSCIUSA') #MSCI US Index daily prices



                            #3: IMPORT SYSTEMIC RISK MAESURES AND RUN MODULES

    #Mahalanobis Distance
SRM_mahalanobis= srm.MahalanobisDist(returns)#define Mahalanobis Distance Formula
Table_1= srm.MahalanobisDist_Table1(returns)[0]

    #Correlation Surprise
SRM_correlationsurprise= srm.Correlation_Surprise(returns)#define Correlation Surprise Score
Correlation_Surprise_Exhibit_5= srm.Correlation_Surprise_Table_Exhbit5(SRM_correlationsurprise)
Correlation_Surprise_Exhibit_6= srm.Correlation_Surprise_Table_Exhbit6(SRM_correlationsurprise, Correlation_Surprise_Exhibit_5)

    #Absorption Ratio
SRM_absorptionratio= srm.Absorption_Ratio(FamaFrench49)#define Absorption Ratio
Absorption_Ratio_Standardised_Shift= srm.Absorption_Ratio_Standardised_Shift(SRM_absorptionratio)


    #Systemic Risk Measures 

systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio] # group systemic risk measures
srm.print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES)




#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
 #   fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
  #  fig.savefig("{}.jpg".format(sysRiskMeasure))

















