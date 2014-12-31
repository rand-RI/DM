                             #1: IMPORT LIBRARY
import pandas.io.data as pdio      
import systemicRiskMeasures as srm                                                 #import Systemic Risk Measures library
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt 
import numpy as np

                            #2: IMPORT DATA

    #Mahalanobis Distance
#Data
Historical_Prices= pd.load('returns')                                              #Import Hisorical Prices for six asset-class indices: US stocks, non-US Stocks, US Bonds, non-US bonds, commodities, and US real estate
Adjusted_Close_Prices= Historical_Prices['Adj Close'].dropna()                     #Extract Adjusted Close prices from Historial Prices of the six assets 
returns= srm.logreturns(Adjusted_Close_Prices)                                     #Convert Adjusted Close Prices from daily returns to Logarithmic returns

returns_Figure5= pd.load('returnsMD_Figure5')                                       # Import Daily returns of World Equities, US small-capitalisation premium(Small-Large), growth Premium(Growth-Value) and HFRI Fund of Funds composite Index

Table_1_returns= pd.load('Table_1') 




    #Correlation Surprise
returns_Figure5= pd.load('returnsMD_Figure5')                                      #need to gather data for this one(Datastream) #US equities MSCI secotors
returns_Figure5= pd.load('returnsMD_Figure5')                                      #European Equities

    
    #Absorption Ratio
FamaFrench49= pd.load('FenFrench49')                                                #51 sectors
MSCIUS_PRICES= pd.load('MSCIUSA')                                                   #MSCI US Index daily prices



                            #3: IMPORT SYSTEMIC RISK MAESURES AND RUN MODULES

    #Mahalanobis Distance
SRM_mahalanobis= srm.MahalanobisDist(returns)                                       #define Mahalanobis Distance Formula
Table_1= srm.MahalanobisDist_Table1(Table_1_returns)[0]

    #Correlation Surprise
SRM_correlationsurprise= srm.Correlation_Surprise(returns)                          #define Correlation Surprise Score
Correlation_Surprise_Exhibit_5= srm.Correlation_Surprise_Table_Exhbit5(SRM_correlationsurprise)
Correlation_Surprise_Exhibit_6= srm.Correlation_Surprise_Table_Exhbit6(SRM_correlationsurprise, Correlation_Surprise_Exhibit_5)

    #Absorption Ratio
SRM_absorptionratio= srm.Absorption_Ratio(FamaFrench49)                             #define Absorption Ratio
Absorption_Ratio_Standardised_Shift= srm.Absorption_Ratio_Standardised_Shift(SRM_absorptionratio)


    #Systemic Risk Measures 
systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio]   # group systemic risk measures
srm.print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES)




#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
 #   fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
  #  fig.savefig("{}.jpg".format(sysRiskMeasure))

















