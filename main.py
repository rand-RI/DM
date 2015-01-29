 
""" This Python scripts is used to generate systemic risk returns from systemic risk modules
    imported from systemicRiskMeasures as srm  in or each Systemic Risk Paper
                                                                                """


"""STAGE 1: 
IMPORT LIBRARY"""
#-------------------------
import pandas as pd
import systemicRiskMeasures as srm                                             #Import Systemic Risk Measures library
#----------------------------------------------------------------------------------------------------------------------------------


"""STAGE 2: 
IMPORT DATA"""
#--------------------------
"""Mahalanobis Distance"""
    #Historical Turbulence Index Calcualted Calculated as Monthly Returns:
Historical_Prices= pd.load('returns')                                          #Import Historical Prices for six asset-class indices: US stocks, non-US Stocks, US Bonds, non-US bonds, commodities, and US real estate
Adjusted_Close_Prices= Historical_Prices['Adj Close'].dropna()                 #Extract Adjusted Close prices from Historial Prices of the six assets 
returns= srm.logreturns(Returns=Adjusted_Close_Prices)                         #Convert Adjusted Close Prices from daily returns to Logarithmic returns

    #Persistence of Turbulence Markets 
returns_Figure5= pd.load('returnsMD_Figure5')                                  #Import Daily returns of World Equities, US small-capitalisation premium(Small-Large), growth Premium(Growth-Value) and HFRI Fund of Funds composite Index
Global_Assets= pd.load('Table1_Global_Assets')    
US_Assets= pd.load('Table1_US_Assets')    
Currency= pd.load('Table1_Currency')    
Table_1_returns= Global_Assets,US_Assets,Currency                               #Import Table_1 returns 

    #Efficent Portfolios, Expected Returns, and Two Estimates of Risk
Table_2_Asset_Classes= pd.load('Table2_Asset_Classes')                          #Import Table_2 Asset Classes
WeightsC=[.2286, .1659, .4995, .0385]
WeightsM=[.3523, .2422, .3281, .0259]  
WeightsA=[.4815, .3219, .1489, .0128] 
portfolio_weights=WeightsC,WeightsM,WeightsA

    #Modified Mean-Variance Optimization:
equilibrium_Weights=[0.1,0.6,0.3]
Equilbrium_returns= returns[['EFA','TIP','VNQ']]
S_P500= pd.load('S&P_500')


#Latex Figures
EM_Asia_xIndia= pd.load('EM_Asia_xIndia')
EM_Asia_xIndia_returns= srm.logreturns(Returns=EM_Asia_xIndia)

US_sectors= pd.load('USsectors')
US_sectors_returns= srm.logreturns(Returns=US_sectors)

UK_sectors= pd.load('UKsectors')
UK_sectors_returns= srm.logreturns(Returns=UK_sectors)

JPN_sectors= pd.load('JPNsectors')
JPN_sectors_returns= srm.logreturns(Returns=JPN_sectors)

CAN_sectors= pd.load('CANsectors')
CAN_sectors_returns= srm.logreturns(Returns=CAN_sectors)
#-------------------------

"""Correlation Surprise"""
    #Import US_Equities, Euro_Equities and Currency data
Exhibit5_USEquities= pd.load('Exhibit5_US_Equities')        #Import Correlation Surprise Exhibit5 US Equities
Exhibit5_Euro_Equities=pd.load('Exhibit5_Euro_Equities')#Import Correlation Surprise Exhibit5 European Equities
Exhibit5_Currency=pd.load('Exhibit5_Currency')             #Import Correlation Surprise Exhibit5 Currency
#-------------------------

"""Absorption Ratio"""
    #Import sectors of the US economy
FamaFrench49= pd.load('FenFrench49')                                           #51 sectors
MSCIUS_PRICES= pd.load('MSCIUSA')                                              #MSCI US Index daily prices

    #Exhibit_9 imported data for a dynamic trading strategy in which has exposure to government bonds and stocks
Treasury_bonds= pd.load('Treasury_Bonds_AR_Exhibit_9')['Adj Close']
MSCIUS_PRICES= pd.load('MSCIUSA') 
#-------------------------
#----------------------------------------------------------------------------------------------------------------------------------


"""STAGE 3: 
IMPORT SYSTEMIC RISK MEASURES AND RUN SIGNALS"""
#-------------------------

"""Mahalanobis Distance"""
        #Input
MD_input=US_sectors_returns           #Change this value for data required
        #Run
SRM_mahalanobis= srm.MahalanobisDist(Returns=MD_input)   
SRM_mahalanobis_turbulent_nonturbulent_days= srm.MahalanobisDist_Turbulent_Returns(MD_returns= SRM_mahalanobis, Returns=MD_input)
MD_input= MD_input.drop('MD',1)
US_sectors_returns  = US_sectors_returns.drop('MD',1)    
        #Graph
SRM_HistoricalTurbulenceIndexGraph= srm.HistoricalTurbulenceIndexGraph( Mah_Days=SRM_mahalanobis,  width=30, figsize=(10,2.5))
#-------------------------

"""Correlation Surprise"""
        #Input
Corr_Input= EM_Asia_xIndia_returns
        #Run
SRM_Correlation_Surprise=srm.Correlation_Surprise(Returns=Corr_Input)
#-------------------------

"""Absorption Ratio"""
        #Input
AR_input= US_sectors_returns 
Comparision_input= US_sectors_returns['FTSE USA H/C EQ & SVS - PRICE INDEX']
# MSCIUS_PRICES #must be same length as AR
        #Run
SRM_absorptionratio= srm.Absorption_Ratio(Returns= AR_input)                        #define Absorption Ratio
#SRM_Absorption_Ratio_Standardised_Shift= srm.Absorption_Ratio_Standardised_Shift(AR_Returns= SRM_absorptionratio[0])
#SRM_Absorption_Ratio_Standardised_Shift_monthly= SRM_Absorption_Ratio_Standardised_Shift.resample('M')
        #Graphs
halflife=0
SRM_AR_plot= SRM_absorptionratio.plot(figsize=(10,4))
#SRM_Absorption_Ratio_and_Stock_Prices_Graph= srm.Absorption_Ratio_VS_MSCI_Graph(MSCI=Comparision_input, AR_returns=SRM_absorptionratio[0])
#SRM_AR_vs_Market_plot= srm.plot_AR(AR=SRM_absorptionratio)
#-------------------------
#----------------------------------------------------------------------------------------------------------------------------------

"""Stage4: 
RUN Empirical Analysis"""
#-------------------------
"""Mahalanobis Distance"""
#SRM_Persistence_of_Turbulence= srm.MahalanobisDist_Table1(Market_Returns=Table_1_returns)
#SRM_Efficient_Portfolios=srm.MahalanobisDist_Table2(Asset_Class= Table_2_Asset_Classes, Weights=portfolio_weights) #need to add [0] to get Table
#SRM_VaR_and_Realised_Returns = srm.MahalanobisDist_Table3(Portfolios=SRM_Efficient_Portfolios[1], beta=0.01)
#SRM_Modified_Mean_Variance_Optimization= srm.MahalanobisDist_Table4(portfolio=Equilbrium_returns, weights=equilibrium_Weights)

SRM_Mean_Var= srm.MahalanobisDist_Table4(portfolio=Equilbrium_returns, weights=equilibrium_Weights)
#SRM_shrink_cov=  srm.shrinking_cov(Market_Portfolio=Equilbrium_returns*equilibrium_Weights,Regress_test= returns.iloc[:,4:5] )
#-------------------------
    
    
"""Correlation Surprise"""
    #Conditional_ave_magn_sur_on_day_of_the_reading= srm.Conditional_ave_magn_sur_on_day_of_the_reading(Exhibit5_USEquities, Exhibit5_Euro_Equities, Exhibit5_Currency)[0] # Import Exhibit 5
    #Correlation_Surprise_Exhibit_6= srm.Conditional_ave_magn_sur_on_day_after_reading(Exhibit5_USEquities, Exhibit5_Euro_Equities, Exhibit5_Currency)
#-------------------------

"""Absorption Ratio """
    #AR_and_Drawdowns= srm.Absorption_Ratio_and_Drawdowns(delta_AR=SRM_Absorption_Ratio_Standardised_Shift)
#Exhibit_9= srm.Exhbit_9(Treasury_bonds=Treasury_bonds, MSCIUS_PRICES= MSCIUS_PRICES)
   
     #absorption Ratio_2 
#SRM_Centraility= srm.AR_systemic_importance(AR=SRM_absorptionratio)
   
#-------------------------    

#PORBIT MODEL

  #Systemic Risk Measures 
#systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio]# group systemic risk measures
#srm.print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES)
#----------------------------------------------------------------------------------------------------------------------------------