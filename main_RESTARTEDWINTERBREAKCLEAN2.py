 
""" This Python scripts is used to generate systemic risk returns from systemic risk modules
    imported from systemicRiskMeasures as srm  in or each Systemic Risk Paper
                                                                                """


"""STAGE 1: 
IMPORT LIBRARY"""
#-------------------------
import pandas as pd
import systemicRiskMeasures1 as srm   
import matplotlib.pyplot as plt    
                                      #Import Systemic Risk Measures library
#----------------------------------------------------------------------------------------------------------------------------------


"""STAGE 2: 
IMPORT DATA"""
#--------------------------
US_sectors= pd.load('USsectors')
US_sectors_returns= srm.logreturns(Returns=US_sectors)
FmmaFrench49_from1926=(pd.load('FF49_1926').resample('M'))[607+155:]
Recession_Values= pd.load('USARECM')
#-------------------------
#----------------------------------------------------------------------------------------------------------------------------------

"""STAGE 3: 
IMPORT SYSTEMIC RISK MEASURES AND RUN SIGNALS"""
#-------------------------
Inputs=FmmaFrench49_from1926
#Input= US_sectors_returns.resample('M') #input monthly returns

"""Mahalanobis Distance"""
        #Input
MD_input=Inputs         #Change this value for data required
        #Run
SRM_mahalanobis= srm.MahalanobisDist(Returns=MD_input)[41:]   
SRM_mahalanobis_turbulent_nonturbulent_days= srm.MahalanobisDist_Turbulent_Returns(MD_returns= SRM_mahalanobis, Returns=MD_input)
                    #drop inputs
Inputs=Inputs.drop('MD',1)
MD_input= MD_input.drop('MD',1)
       #Graph
srm.HistoricalTurbulenceIndexGraph( Mah_Days=SRM_mahalanobis,  width=30, figsize=(10,2.5), datesize='M')
#-------------------------
"""Correlation Surprise"""
        #Input
Corr_Input= Inputs
        #Run
SRM_Correlation_Surprise=srm.Correlation_Surprise(Returns=Corr_Input)  #need to reshift by 41 months
        #Graph
srm.Corr_plot(Corr_sur=SRM_Correlation_Surprise[0][41:], Mag_sur=SRM_Correlation_Surprise[1][41:],  width=25, figsize=(10,4.5), datesize='M')
#-------------------------
"""Absorption Ratio"""
        #Input
AR_input= Inputs
SRM_absorptionratio= srm.Absorption_Ratio(Returns= AR_input, halflife=int(500/12))                        #define Absorption Ratio
    #Graphs
srm.plot_AR(AR=SRM_absorptionratio, figsize=(10,2.5),yaxis=[0.75,0.9])
#-------------------------
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
        #Import Same Data for Comparision 
Input_returns=Inputs[:294]    #Start 1990-01-31
VIX_returns= pd.load('^VIX').resample('M') [:294]             
Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M')[155:456]
Recession_Values= Recession_Values[483:]
    #List of Equities and Fixed Income Switch Stradegy Portfolio
#**** Note the Balanced_port in this case has an index len one month ahead due to monthly returns not yet out but daily returns of the month so far have been calculated(This allows the index to purchase returns from the day after the predicted monthly returns)
#Set Parameters 
Window_Range= 101   #Must be greater than 41 as Absorption Ratio requires 500day rolling window. Therefore the Window size is Window-41                                                           
Forecast_Range=len(Input_returns)-Window_Range                 #Months
#---------------------------------------------------------
#--------------------------
    #  2: RUN PROBIT

#http://arch.readthedocs.org/en/latest/bootstrap/bootstrap_examples.html
Probit_Forecast=pd.DataFrame()
for i in range(Forecast_Range):
    window= int(Window_Range) #Create Intial window size(eg First 2500 Days)
    Input= Input_returns[0:window+i]
    VIX= VIX_returns[0:window+i]  
    Recession_data= Recession_Values[0:window+i] #Set Input
    Recession_data=pd.DataFrame(Recession_data.values, index=VIX.index)    
    Probit_function=srm.Probit(Input_Returns=Input, vix=VIX, recession_data=Recession_data)  #Generate Probit Parameters 
    
    Intercept= Probit_function[0][0]                                                       
    First_coeff= Probit_function[0][1]              
    Second_coeff= Probit_function[0][2]
    Third_coeff= Probit_function[0][3]
    Fourth_coeff= Probit_function[0][4]
    Input_first_variable=Probit_function[1]['MD'].tail()[4]
    Input_second_varibale=Probit_function[1]['Mag_Corr'].tail()[4]
    Input_third_varibale=Probit_function[1]['AR'].tail()[4]
    Input_fourth_varibale=Probit_function[1]['VIX'].tail()[4]
    Function= Intercept+ First_coeff*Input_first_variable + Second_coeff*Input_second_varibale + Third_coeff*Input_third_varibale +Fourth_coeff*Input_fourth_varibale #Create Probit Function and generate Forecast Value
    
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
#---------------------------
# Optimisations 
opt_returns=[]
for a in range(30):
    g=a   #look back window
    Theshold=Initial_Theshold
    for i in range(0,len(Probit_Forecast)-1-g):   #need to make sure it finishes at end and starts at orgin    
        """ What you will get"""
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
    Switch_Portfolio_results=Switch_Portfolio.sum().sum() #grabs theshold optimised portfolio
    opt_returns.append(Switch_Portfolio_results)
    Switch_Portfolio=pd.DataFrame() 
    print ["Iteration Completed",a]
#Get op window
maximum_win= np.max(opt_returns)
max_loc_in_range_= opt_returns.index(maximum_win)
Theshold_window=max_loc_in_range_
#--------------------------------------------------------
#Actual
Probit_Forecast=Probit_Forecast #Set Probit Forecasts for the previous input of US monthly returns
Rebalanced_portfolio= Balanced_port[Window_Range:]      #Set the Portfolio of Equities and Bonds at same starting date as Probit
Switch_Portfolio=pd.DataFrame()                     #Define Switch_Portfolio as empty dataframe to append values later in loops below
Theshold_Values=[]              #Set empty Theshold Value to append values form loop below
Returns_that_you_will_get=[]    #"" "" ""
Initial_Theshold=0   
Theshold=Initial_Theshold
g=Theshold_window 
for i in range(0,len(Probit_Forecast)-1-g):   #need to make sure it finishes at end and starts at orgin    
    """ What you will get"""
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
plt.plot(Probit_Forecast[1:].index,Theshold_Values ) #need to append extra set #starts at 0 for 12month
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
plt.plot(Net_returns_RP.index[:189],Net_returns_RP.values[:189], label='S&P 500')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
plt.grid()
plt.show()
