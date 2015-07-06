 
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
"""
Start='19770228'
Start_Recession_Values='19770201'
End='20140630' #20140630 latest St Louis Recession data date
window_range= 120 #months

FmmaFrench49_from1926=(pd.load('FF49_1926').resample('M',how='sum')).loc[Start:End]   #1926
Recession_Values= pd.load('USARECM')     #1949-10-01
Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M',how='sum').loc[Start:End]
Recession_Values= Recession_Values[Start_Recession_Values:]  
"""


Start='19930131'
Start_Recession_Values='19930101'
End='20140630' #20140630 latest St Louis Recession data date
window_range= 60  #months

FmmaFrench49_from1926=(pd.load('FF49_1926').resample('M',how='sum')).loc[Start:End]
Recession_Values= pd.load('USARECM')
VIX_returns= pd.load('^VIX').resample('M').loc[Start:End] #only goes back to 1990     
Balanced_port= srm.logreturns(Returns=pd.load('Probit_portfolio')).resample('M',how='sum').loc[Start:End]
Recession_Values= Recession_Values[Start_Recession_Values:] 

#-------------------------
#----------------------------------------------------------------------------------------------------------------------------------
"""STAGE 3: 
IMPORT SYSTEMIC RISK MEASURES AND RUN SIGNALS"""
#-------------------------
Input=FmmaFrench49_from1926
Inputs=Input[window_range:]
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
"""
# ----------------------------------
   #  1: INPUTS
        #Import Same Data for Comparision 
Input_returns=Input    #Start 1990-01-31
VIX_returns= VIX_returns         
Balanced_port= Balanced_port
Recession_Values= Recession_Values  
Window_Range= 41+window_range       #10 year window   #Must be greater than 41 as Absorption Ratio requires 500day rolling window. Therefore the Window size is Window-41                                                           
Forecast_Range=len(Input_returns)-Window_Range +1               #Months
#---------------------------------------------------------
#--------------------------
    #  2: RUN PROBIT
#http://arch.readthedocs.org/en/latest/bootstrap/bootstrap_examples.html
Probit_Forecast=pd.DataFrame()
for i in range(Forecast_Range):
    window= int(Window_Range) 
    Input= Input_returns[0:window+i]
    VIX= VIX_returns[0:window+i]  
    Recession_data= Recession_Values[0:window+i] #Set Input
    Recession_data=pd.DataFrame(Recession_data.values, index=Input.index)    
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
    Function= Intercept+ First_coeff*Input_first_variable + Second_coeff*Input_second_varibale + Third_coeff*Input_third_varibale +Fourth_coeff*Input_fourth_varibale 
    #Create Probit Function and generate Forecast Value
    
    df=pd.DataFrame(index=(Input_returns[0:window+i+1].tail()[4:].index)) #Appending month ahead at the moment    
    df['Probit']=Function
    Probit_Forecast=Probit_Forecast.append(df) 
    print ['Probit Iteration', i, 'Out of', Forecast_Range-1]
#When Probit is for that month grab that month...it is always forecasted ahead and will be appended with the Forecast month's results
#Therefore when chosing ...Probit is a certain value chose the month that it is appended to(which is the forecast) as this will take the results from that previous point
df=pd.DataFrame(index=((Input_returns[0:window+i+1].tail()[4:].index)+1)) #Appending month ahead at the moment    
df['Probit']=Function
Probit_Forecast=Probit_Forecast[0:len(Probit_Forecast)-1]
Probit_Forecast=Probit_Forecast.append(df) 
#---------------------------------------------------------
#----------------------------------
    #  3  Generating Switching Portfolfio Stradgy in combination with Threshold calculations
#1: Set Parameters
Probit_Forecast=Probit_Forecast[Probit_Forecast>-10].fillna(0) #outlier
 #Set Probit Forecasts for the previous input of US monthly returns
Rebalanced_portfolio= Balanced_port[Window_Range:]      #Set the Portfolio of Equities and Bonds at same starting date as Probit
Switch_Portfolio=pd.DataFrame()                     #Define Switch_Portfolio as empty dataframe to append values later in loops below
Theshold_Values=[]              #Set empty Theshold Value to append values form loop below
Returns_that_you_will_get=[]    #"" "" ""
Initial_Theshold=0                  #Let intial theshold equal 0          
Theshold=Initial_Theshold
for i in range(0,len(Probit_Forecast)-1):   #need to make sure it finishes at end and starts at orgin    
    """ What you will get"""
    Predicted_Probit_decider=Probit_Forecast['Probit'][i:i+1][0]                    #Grabs First Row
    Theshold=Theshold
    if (Predicted_Probit_decider>Theshold):   
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[i:1+i].ix[:,1:2]) #Fixed Income
    else:
        Switch_Portfolio=Switch_Portfolio.append(Rebalanced_portfolio[i:1+i].ix[:,0:1])  #Equity 
    Switch_Portfolio=Switch_Portfolio.fillna(0)
    Returns_that_you_will_get.append(Switch_Portfolio.sum().sum())
       #-----------------------------------------
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
plt.grid()
plt.show()
#----------------------------------
#----------------------------------

"""IF $100 Invested""" #(will need to add transaction costs )
Switch_portfolio_total_returns=(Switch_Portfolio['^GSPC']+Switch_Portfolio['^TYX'])+1
Net_return_values_SP=[]
Initial_Amount=100
for i in range(1,len(Switch_portfolio_total_returns)+1):
    Net_return=(Switch_portfolio_total_returns[i-1:i]*Initial_Amount)[0]
    Net_return_values_SP.append(Net_return)
    Initial_Amount=Net_return
Net_returns_SP=pd.DataFrame(Net_return_values_SP,index=Switch_portfolio_total_returns[0:].index)

SP500=Rebalanced_portfolio['^GSPC']+1
Net_return_values_R=[]
Initial_Amount=100
for i in range(1,len(SP500)+1):
    Net_return=(SP500[i-1:i]*Initial_Amount)[0]
    Net_return_values_R.append(Net_return)
    Initial_Amount=Net_return
Net_returns_R=pd.DataFrame(Net_return_values_R,index=SP500.index)

fig= plt.figure(1, figsize=(10,3))
plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
plt.xlabel('Year')                                                          #label x axis Year
plt.ylabel('Price')                                                         #label y axis Index
plt.suptitle('Comparision of Returns',fontsize=12)   
plt.plot(Net_returns_SP.index,Net_returns_SP.values, label='Switch Strad', linestyle='--')
plt.plot(Net_returns_R.index,Net_returns_R.values, label='S&P 500')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
plt.grid()
plt.show()
print ['Over Index of', Net_returns_R.index]



















"""
#Cumaltive sum Graph
Switch_portfolio_total_returns=Switch_Portfolio['^GSPC']+Switch_Portfolio['^TYX']
Net_return_values_SP=[]
for i in range(1,len(Switch_portfolio_total_returns)+1):
    Net_return=Switch_portfolio_total_returns[0:i].sum()
    Net_return_values_SP.append(Net_return)
Net_returns_SP=pd.DataFrame(Net_return_values_SP,index=Switch_portfolio_total_returns[0:].index)

SP500=Rebalanced_portfolio['^GSPC']
Net_return_values_RP=[]
for i in range(1,len(SP500)+1):
    Net_return=SP500[0:i].sum()
    Net_return_values_RP.append(Net_return)
Net_returns_RP=pd.DataFrame(Net_return_values_RP,index=SP500.index)   

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
print ['Over Index of', Net_returns_RP.index]
"""


















  #List of Equities and Fixed Income Switch Stradegy Portfolio
#**** Note the Balanced_port in this case has an index len one month ahead due to monthly returns not yet out but daily returns of the month so far have been calculated(This allows the index to purchase returns from the day after the predicted monthly returns)
#Set Parameters 