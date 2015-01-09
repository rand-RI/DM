 
""" This Python scripts is used to generate systemic risk returns from systemic risk modules
    imported from systemicRiskMeasures as srm  in or each Systemic Risk Paper
                                                                                """


                #STAGE 1: IMPORT LIBRARY
import systemicRiskMeasures as srm                                             #Import Systemic Risk Measures library
import pandas as pd


                #STAGE 2: IMPORT DATA
    #Mahalanobis Distance
Historical_Prices= pd.load('returns')                                          #Import Historical Prices for six asset-class indices: US stocks, non-US Stocks, US Bonds, non-US bonds, commodities, and US real estate
Adjusted_Close_Prices= Historical_Prices['Adj Close'].dropna()                 #Extract Adjusted Close prices from Historial Prices of the six assets 
returns= srm.logreturns(Returns=Adjusted_Close_Prices)                         #Convert Adjusted Close Prices from daily returns to Logarithmic returns

returns_Figure5= pd.load('returnsMD_Figure5')                                  #Import Daily returns of World Equities, US small-capitalisation premium(Small-Large), growth Premium(Growth-Value) and HFRI Fund of Funds composite Index

Table_1_returns= pd.load('Table_1')                                            #Import Table 1 returns for Global Assets, US assets, US sectors, Currencies, US fixed income, US Treasury notes and US credit

    #Correlation Surprise
Exhibit5_USEquities= pd.load('CorrelationSurprise_Exhibit5_USEquities')        #Import Correlation Surprise Exhibit5 US Equities
Exhibit5_EuropeanEquities=pd.load('CorrelationSurprise_Exhibit5_EuropeanEquities')#Import Correlation Surprise Exhibit5 European Equities
Exhibit5_Currency=pd.load('CorrelationSurprise_Exhibit5_Currency')             #Import Correlation Surprise Exhibit5 Currency

    
    #Absorption Ratio
FamaFrench49= pd.load('FenFrench49')                                           #51 sectors
MSCIUS_PRICES= pd.load('MSCIUSA')                                              #MSCI US Index daily prices



                #STAGE 3: IMPORT SYSTEMIC RISK MAESURES AND RUN MODULES
    #Mahalanobis Distance
SRM_mahalanobis= srm.MahalanobisDist(Returns=returns)                          #Import Mahalanobis Distance Formula given the input of returns
Table_1= srm.MahalanobisDist_Table1(Returns=Table_1_returns)[0]                #Import Table 2 data
Table_2=srm.MahalanobisDist_Table2(Returns=returns, Mahalanobis_Distance_Returns=srm.MahalanobisDist(Returns=returns))#Need to change Returns= Table_2 returns

    #Correlation Surprise
SRM_correlationsurprise= srm.Correlation_Surprise(Returns=returns)             #define Correlation Surprise Score
Correlation_Surprise_Exhibit_5= srm.Correlation_Surprise_Table_Exhbit5(Exhibit5_USEquities, Exhibit5_EuropeanEquities, Exhibit5_Currency) # Import Exhibit 5
#Correlation_Surprise_Exhibit_6= srm.Correlation_Surprise_Table_Exhbit6(SRM_correlationsurprise, Correlation_Surprise_Exhibit_5) #Import Exhibit 6

    #Absorption Ratio
SRM_absorptionratio= srm.Absorption_Ratio(FamaFrench49)                        #define Absorption Ratio
Absorption_Ratio_Standardised_Shift= srm.Absorption_Ratio_Standardised_Shift(SRM_absorptionratio)


    #Systemic Risk Measures 
systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio]# group systemic risk measures
srm.print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES)





#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
 #   fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
  #  fig.savefig("{}.jpg".format(sysRiskMeasure))

















