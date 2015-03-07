 
""" This Python scripts is used to generate systemic risk returns from systemic risk modules
    imported from systemicRiskMeasures as srm  in or each Systemic Risk Paper
                                                                                """


"""STAGE 1: 
IMPORT LIBRARY"""
#-------------------------
import pandas as pd
import systemicRiskMeasures as srm   
import matplotlib.pyplot as plt    
                                      #Import Systemic Risk Measures library
#----------------------------------------------------------------------------------------------------------------------------------


"""STAGE 2: 
IMPORT DATA"""
#--------------------------
"""Mahalanobis Distance"""
    #Historical Turbulence Index Calcualted Calculated as Monthly Returns:
"""
Historical_Prices= pd.load('returns')                                          #Import Historical Prices for six asset-class indices: US stocks, non-US Stocks, US Bonds, non-US bonds, commodities, and US real estate
Adjusted_Close_Prices= Historical_Prices['Adj Close'].dropna()                 #Extract Adjusted Close prices from Historial Prices of the six assets 
returns= srm.logreturns(Returns=Adjusted_Close_Prices)                         #Convert Adjusted Close Prices from daily returns to Logarithmic returns

    #Persistence of Turbulence Markets 
returns_Figure5= pd.load('returnsMD_Figure5')                                  #Import Daily returns of World Equities, US small-capitalisation premium(Small-Large), growth Premium(Growth-Value) and HFRI Fund of Funds composite Index
Global_Assets= pd.load('Table1_Global_Assets') 
Global_Assets_log= srm.logreturns(Returns=Global_Assets)   
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
equilibrium_Weights=[0.4,0.6]
#Equilbrium_returns= returns[['EFA','TIP','VNQ']]
#S_P500= pd.load('S&P_500')
#S_P500_returns= srm.logreturns(Returns=S_P500)
Equilbrium_returns= pd.load('Equilibrium_returns')
Equilbrium_returns= srm.logreturns(Returns=Equilbrium_returns) *equilibrium_Weights


#Latex Figures
EM_Asia_xIndia= pd.load('EM_Asia_xIndia')
EM_Asia_xIndia_returns= srm.logreturns(Returns=EM_Asia_xIndia)
"""
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
Input= US_sectors_returns.resample('M') #input monthly returns

"""Mahalanobis Distance"""
        #Input
MD_input=Input           #Change this value for data required
        #Run
SRM_mahalanobis= srm.MahalanobisDist(Returns=MD_input)   
SRM_mahalanobis_turbulent_nonturbulent_days= srm.MahalanobisDist_Turbulent_Returns(MD_returns= SRM_mahalanobis, Returns=MD_input)
                    #drop inputs
Input=Input.drop('MD',1)
MD_input= MD_input.drop('MD',1)
       #Graph
SRM_HistoricalTurbulenceIndexGraph= srm.HistoricalTurbulenceIndexGraph( Mah_Days=SRM_mahalanobis,  width=30, figsize=(10,2.5), datesize='M')
#-------------------------

"""Correlation Surprise"""
        #Input
Corr_Input= Input
        #Run
SRM_Correlation_Surprise=srm.Correlation_Surprise(Returns=Corr_Input)
        #Graph
srm.Corr_plot( Corr_sur=SRM_Correlation_Surprise[0], Mag_sur=SRM_Correlation_Surprise[1],  width=25, figsize=(10,4.5), datesize='M')
#-------------------------

"""Absorption Ratio"""
        #Input
AR_input= Input
#Comparision_input= US_sectors_returns['FTSE USA H/C EQ & SVS - PRICE INDEX']
# MSCIUS_PRICES #must be same length as AR
        #Run
SRM_absorptionratio= srm.Absorption_Ratio(Returns= AR_input, halflife=int(500/12))                        #define Absorption Ratio
#SRM_Absorption_Ratio_Standardised_Shift= srm.Absorption_Ratio_Standardised_Shift(AR_Returns= SRM_absorptionratio)
#SRM_Absorption_Ratio_Standardised_Shift_monthly= SRM_Absorption_Ratio_Standardised_Shift.resample('M')
        #Graphs
SRM_AR_plot= srm.plot_AR(AR=SRM_absorptionratio, figsize=(10,2.5),yaxis=[0.84,0.9])
#SRM_Absorption_Ratio_and_Stock_Prices_Graph= srm.Absorption_Ratio_VS_MSCI_Graph(MSCI=Comparision_input, AR_returns=SRM_absorptionratio[0])
#SRM_AR_vs_Market_plot= srm.plot_AR(AR=SRM_absorptionratio)
#-------------------------
#----------------------------------------------------------------------------------------------------------------------------------

"""Stage4: 
RUN Empirical Analysis"""         #THE HASTAGGED OUT LINES BELOW ARE DUE TO SLOW FOR LOOPS, HOWEVER, ALL WORK CORRECTLY  
#-------------------------
"""Mahalanobis Distance"""
#SRM_Persistence_of_Turbulence= srm.MahalanobisDist_Table1(Market_Returns=Table_1_returns)
#SRM_Efficient_Portfolios=srm.MahalanobisDist_Table2(Asset_Class= Table_2_Asset_Classes, Weights=portfolio_weights) #need to add [0] to get Table
#SRM_VaR_and_Realised_Returns = srm.MahalanobisDist_Table3(Portfolios=SRM_Efficient_Portfolios[1], beta=0.01)
#SRM_Mean_Var= srm.Mod_Mean_Var(portfolio=Equilbrium_returns, full_trailing=returns)   
#SRM_=srm.Mod_Mean_Var_Exp_Wind(sample=returns, market_portfolio=returns)

#Mean_Var only generates the optimal values for the conditional portfolio ....just need to find a market portfolio of N assets to generate unconditoned results
#Need
#-------------------------
    
    
"""Correlation Surprise"""
#Conditional_ave_magn_sur_on_day_of_the_reading= srm.Conditional_ave_magn_sur_on_day_of_the_reading(Exhibit5_USEquities, Exhibit5_Euro_Equities, Exhibit5_Currency)[0] # Import Exhibit 5
#Correlation_Surprise_Exhibit_6= srm.Conditional_ave_magn_sur_on_day_after_reading(Exhibit5_USEquities, Exhibit5_Euro_Equities, Exhibit5_Currency)
#-------------------------

"""Absorption Ratio """
#SRM_AR_all= srm.plot_AR_ALL(US=US_sectors_returns, UK=UK_sectors_returns, JPN=JPN_sectors_returns, halflife=250)
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


"""Stage5: 
Probit Model"""  
#----------------------------------
""" 
This model uses monthly returns to generate a forecasting Probit model

Window_Range is used to set the initial training sample
Forecast_Range is used to iterate over data commencing after Window_Range

The first for_loop is used to generate the Probit Forecast containing the
three variables MD,MSCS and AR.
For any month(t), it is concluded systemic if either two of the three variables are satisfied.
Rules for each variables include
MD= Top 80 Percentile
MGCS= Top 80 Percentile
AR= Top 80 Percentile 

The second loop is used to generate the returns received from switching in and out of equities 
and fixed income during periods of heighten systemic risk.
When the Probit_Forecast is >0 the switching stradgy contains 100% Fixed income and when
Probit_Forecast is <0 the switching stradgy contains 100% equities.
It should be noted that the Balanced_port should be imported with an equal time series
as the Input_returns
"""
# ----------------------------------
   #  1: INPUTS

Input_returns=(pd.load('FF49_1926').resample('M'))[607:]    #Start 1970-01-31
                          
Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M')[:456]
    #List of Equities and Fixed Income Switch Stradegy Portfolio
#**** Note the Balanced_port in this case has an index len one month ahead due to monthly returns not yet out but daily returns of the month so far have been calculated(This allows the index to purchase returns from the day after the predicted monthly returns)
#Set Parameters 
Window_Range=200                                   #                                                           
Forecast_Range=len(Input_returns)-Window_Range                 #Months


#Input_returns=US_sectors_returns.resample('M')      #Input Times Series of Assets wanted to be tested against Sysmteic Risk Rules and Probit Model                             
#Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M')    #List of Equities and Fixed Income Switch Stradegy Portfolio
#**** Note the Balanced_port in this case has an index len one month ahead due to monthly returns not yet out but daily returns of the month so far have been calculated(This allows the index to purchase returns from the day after the predicted monthly returns)
#Set Parameters 
#Window_Range=120                                    #                                                           
#Forecast_Range=len(Input_returns)-Window_Range                 #Months
#---------------------------------------------------------
#--------------------------
    #  2: RUN PROBIT

Probit_Forecast=pd.DataFrame()
for i in range(Forecast_Range):
    window= int(Window_Range)                                                       #Create Intial window size(eg First 2500 Days)
    Input= Input_returns[0:window+i]                                      #Set Input
    Probit_function=srm.Probit(Input_Returns=Input)                            #Generate Probit Parameters 
    
    Intercept= Probit_function[0][0]                                                       
    First_coeff= Probit_function[0][1]              
    Second_coeff= Probit_function[0][2]
    Third_coeff= Probit_function[0][3]
    Input_first_variable=Probit_function[1]['MD'].tail()[4]
    Input_second_varibale=Probit_function[1]['Mag_Corr'].tail()[4]
    Input_third_varibale=Probit_function[1]['AR'].tail()[4]
    Function= Intercept+ First_coeff*Input_first_variable + Second_coeff*Input_second_varibale + Third_coeff*Input_third_varibale #Create Probit Function and generate Forecast Value
    
    df=pd.DataFrame(index=(Input_returns[0:window+i+1].tail()[4:].index)) #Appending month ahead at the moment    
    df['Probit']=Function
    Probit_Forecast=Probit_Forecast.append(df) 
    print ['Probit Iteration', i]
    
#When Probit is for that month grab that month...it is always forecasted ahead and will be appended with the Forecast month's results
#Therefore when chosing ...Probit is a certain value chose the month that it is appended to(which is the forecast) as this will take the results from that previous point
#---------------------------------------------------------
#----------------------------------

    #  3  Generating Switching Portfolfio Stradgy in combination with Threshold calculations

#1: Set Parameters
#---------------------------
#---------------------------------------------------------
import numpy as np
Probit_Forecast=Probit_Forecast #Set Probit Forecasts for the previous input of US monthly returns
Rebalanced_portfolio= Balanced_port[Window_Range:]      #Set the Portfolio of Equities and Bonds at same starting date as Probit
Switch_Portfolio=pd.DataFrame()                     #Define Switch_Portfolio as empty dataframe to append values later in loops below
Theshold_Values=[]              #Set empty Theshold Value to append values form loop below
Returns_that_you_will_get=[]    #"" "" ""
Initial_Theshold=0                  #Let intial theshold equal 0          
Theshold=Initial_Theshold
#---------------------------

for a in range(100):
    opt_returns=[]
    for i in range(0,len(Probit_Forecast)-1):   #need to make sure it finishes at end and starts at orgin    
        """ What you will get"""
        g=a #look back window 
        Predicted_Probit_decider=Probit_Forecast['Probit'][i+g:i+1+g][0]                    #Grabs First Row
        Theshold=Theshold
        if (Predicted_Probit_decider>Theshold):   
            Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[i+g:1+i+g].ix[:,1:2]) #Fixed Income
        else:
            Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[i+g:1+i+g].ix[:,0:1]) 
        Switch_Portfolio=Switch_Portfolio.fillna(0)
        Returns_that_you_will_get.append(Switch_Portfolio.sum().sum())
       #-----------------------------------------
        """What you should have chosen"""
        Returns=[]
        for k in range(1,100):                  #Set the looking back range
            New_Theshold= Probit_Forecast[i:i+1+g].quantile(k*0.01)[0]
            
            Test_DF=Rebalanced_portfolio[i:i+1+g]
            Test_DF['Probit']=Probit_Forecast[i:i+1+g]
            Test_DF=Test_DF.dropna()
            x=Test_DF[Test_DF['Probit']>New_Theshold]
            y=Test_DF[Test_DF['Probit']<=New_Theshold]
            x=x['^TYX']
            y=y['^GSPC']
            z=pd.DataFrame(index=Test_DF.index)
            z['^GSPC']=y
            z['^TYX']=x
            z=z.fillna(0)
            total_returns=z.sum().sum()
            Returns.append(total_returns)
        maximum= np.max(Returns)
        max_loc_in_range= Returns.index(maximum)
        Theshold=max_loc_in_range+1
        Theshold=Probit_Forecast[i:i+1+g]['Probit'].quantile(Theshold*0.01)
        Theshold_Values.append(Theshold)
        print ["Iteration Completed",i]
    Switch_Portfolio_results=Switch_Portfolio.sum().sum()
    opt_returns.append(Switch_Portfolio_results)
    maximum_= np.max(opt_returns)
    max_loc_in_range_= opt_returns.index(maximum_)
    Theshold=max_loc_in_range_
    Switch_Portfolio=pd.DataFrame() 
    
    
        
        
  
    
    """
    #What you should get next time
    Returns=[]
    Switch_Portfolio_test=pd.DataFrame()
    for k in range(1,100):
        for j in range(i+1):
            New_Theshold= Probit_Forecast[0:i+1].quantile(k*0.01)[0]
            if (Predicted_Probit_decider>New_Theshold):   
                Switch_Portfolio_test=Switch_Portfolio_test.append(Rebalanced_portfolio[1+j:2+j].ix[:,1:2]) #Fixed Income
            else:
                Switch_Portfolio_test=Switch_Portfolio_test.append(Rebalanced_portfolio[1+j:2+j].ix[:,0:1]) 
        Switch_Portfolio_test=Switch_Portfolio_test.fillna(0)
        total_returns=Switch_Portfolio_test.sum().sum()
        Returns.append(total_returns)
    maximum= np.max(Returns)
    max_loc_in_range= Returns.index(maximum)
    Theshold=max_loc_in_range+1
    Theshold=Probit_Forecast[0:i+1]['Probit'].quantile(Theshold*0.01)
    Theshold_Values.append(Theshold)
    #----------------------------------
    """
     #This allows you to see what the portfolio is doing

    
#---------------------------------------------------------
#---------------------------------------------------------
    #  4  RESULTS

#Probit Graph 
fig= plt.figure(1, figsize=(10,3))
plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
plt.xlabel('Year')                                                          #label x axis Year
plt.ylabel('Index')                                                         #label y axis Index
plt.suptitle('Probit Forecast',fontsize=12)   
plt.bar(Probit_Forecast.index,Probit_Forecast.values, width=0.5,color='w')#graph bar chart of Ma
plt.plot(Probit_Forecast[2:].index,Theshold_Values ) #need to append extra set #starts at 0 for 12month
plt.grid()
plt.show()
#----------------------------------
#----------------------------------

#Cumaltive sum Graph
Switch_portfolio_total_returns=Switch_Portfolio['^GSPC']+Switch_Portfolio['^TYX']
Net_return_values_SP=[]
for i in range(2,len(Switch_portfolio_total_returns)):
    Net_return=Switch_portfolio_total_returns[0:i].sum()
    Net_return_values_SP.append(Net_return)
Net_returns_SP=pd.DataFrame(Net_return_values_SP,index=Switch_portfolio_total_returns[2:].index)

SP500=Rebalanced_portfolio['^GSPC']
Net_return_values_RP=[]
for i in range(3,len(SP500)):
    Net_return=SP500[0:i].sum()
    Net_return_values_RP.append(Net_return)
Net_returns_RP=pd.DataFrame(Net_return_values_RP,index=SP500[3:].index)   

fig= plt.figure(1, figsize=(10,3))
plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
plt.xlabel('Year')                                                          #label x axis Year
plt.ylabel('Cumulative Returns')                                                         #label y axis Index
plt.suptitle('Comparision of Returns',fontsize=12)   
plt.plot(Net_returns_SP.index,Net_returns_SP.values, label='Switch Strad', linestyle='--')
plt.plot(Net_returns_RP.index,Net_returns_RP.values, label='S&P 500')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
plt.grid()
plt.show()





"""
import numpy as np
Rebalanced_portfolio= Balanced_port[Window_Range:]
Switch_Portfolio=pd.DataFrame()
Theshold_Values=[]
for i in range(len(Probit_Forecast)):        #iterate 
    grab_next_month= (Probit_Forecast[i:i+1]['Probit'][0])
    #----------------------------------------------------
        #Theshold Calculations
    Returns=[]
    for j in range(1,100):
        if (grab_next_month>(Probit_Forecast[0:i+1]['Probit']).quantile(j*0.01)):
            Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[1+i:2+i].ix[:,1:2]) #Fixed Income
        else:
            Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[1+i:2+i].ix[:,0:1]) 
    
        Switch_Portfolio=Switch_Portfolio.fillna(0)
        total_returns=Switch_Portfolio.sum().sum()
        Returns.append(total_returns)
        Switch_Portfolio= Switch_Portfolio.drop(Switch_Portfolio.index[len(Switch_Portfolio)-1])
        #remove last row of dataframe every test
    maximum= np.max(Returns)
    max_loc_in_range= Returns.index(maximum)
    Theshold=max_loc_in_range+1
    Theshold_Values.append((Probit_Forecast[0:i+1]['Probit']).quantile(Theshold*0.01))
               
    #----------------------------------------------------      
    
    if (grab_next_month>(Probit_Forecast[0:i+1]['Probit']).quantile(Theshold*0.01)):   #4 appears optimal  #If the next month is Systemic Invest 100% in Fixed Income otherwise invest in 100% in Equities
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[1+i:2+i].ix[:,1:2]) #Fixed Income
    else:
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[1+i:2+i].ix[:,0:1]) 
    #Generate next theshold
Switch_Portfolio=Switch_Portfolio.fillna(0)
"""




"""
This Model is used to calculate a new theshold 
import numpy as np
Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M')
Rebalanced_portfolio= Balanced_port[Window_Range:]
Switch_Portfolio=pd.DataFrame()
for i in range(len(Probit_Forecast)):        #iterate 
    grab_next_month= (Probit_Forecast[i:i+1]['Probit'][0])
    #----------------------------------------------------
        #Theshold Calculations
    Returns=[]
    for j in range(1,100):
        #Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M')
        #Rebalanced_portfolio= Balanced_port[Window_Range:]
        x=pd.DataFrame()
        for k in range(i):        #iterate 
            grab_next_month= (Probit_Forecast[k:k+1]['Probit'][0])
            if (grab_next_month>(Probit_Forecast[0:k+1]['Probit']).quantile(j*0.01)):   #4 appears optimal  #If the next month is Systemic Invest 100% in Fixed Income otherwise invest in 100% in Equities
                x=x.append(Rebalanced_portfolio[1+k:2+k].ix[:,1:2]) #Fixed Income
            else:
                x=x.append(Rebalanced_portfolio[1+k:2+k].ix[:,0:1]) 
            #Generate next theshold
        x=x.fillna(0)
        x=x.sum().sum()
        Returns.append(x)
    maximum= np.max(Returns)
    max_loc_in_range= Returns.index(maximum)
    Theshold=max_loc_in_range+1    
    #----------------------------------------------------      
    
    if (grab_next_month>(Theshold*0.01)):   #4 appears optimal  #If the next month is Systemic Invest 100% in Fixed Income otherwise invest in 100% in Equities
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[1+i:2+i].ix[:,1:2]) #Fixed Income
    else:
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[1+i:2+i].ix[:,0:1]) 
    #Generate next theshold
Switch_Portfolio=Switch_Portfolio.fillna(0)
"""
#maybe append first to see what is best


""" 
Rebalanced_portfolio['Probit']=Probit_Forecast
Rebalanced_portfolio=Rebalanced_portfolio.dropna()
x=Rebalanced_portfolio[Rebalanced_portfolio['Probit']>4]
y=Rebalanced_portfolio[Rebalanced_portfolio['Probit']<=4]
x=x['^TYX']
y=y['^GSPC']
z=pd.DataFrame(index=Rebalanced_portfolio.index)
z['^GSPC']=y
z['^TYX']=x
z.fillna(0)
"""
