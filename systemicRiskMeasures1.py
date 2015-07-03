def CreateDataFrameWithTimeStampIndex(DataFrame):
    import datetime 
    import pandas as pd    
    
    New_index=[]
    for i in range(len(DataFrame)):
        timestamp= datetime.datetime.strptime(DataFrame.index[i],'%d/%m/%Y')
        New_index.append(timestamp)
    New_DataFrame=pd.DataFrame(DataFrame.values, index=New_index, columns=DataFrame.columns)    
  
  #--------------------------------------------------------------------------- 
    return New_DataFrame
   #---------------------------------------------------------------------------


def logreturns(Returns):    #GENERATED LOGARITHMIC RETURNS
    
    import numpy as np    
        
    returns = np.log(Returns/Returns.shift(1)).dropna()  #Generate log returns
                                 #Choose if Daily, Monthly, Yearly(ect) dataframe is required
  
  #--------------------------------------------------------------------------- 
    return   returns                                                    #Return Log returns
   #---------------------------------------------------------------------------

                ##Systemic Risk Measures##

#Journal Article: Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
def MahalanobisDist(Returns):                                                  #define MahalanobisDistance function with Returns being a singal dataFrame with n number of columns
  
        #stage1: IMPORT LIBRARIES
    import pandas as pd                                                        #import pandas    
    import numpy as np                                                         #import numpy
    
        #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= Returns.cov()                                           #Generate covariance matrix for historical returns
    return_inverse= np.linalg.inv(return_covariance)                           #Generate inverse covariance matrix for historical returns

        #stage3: CALCULATE THE DIFFERENCE BETWEEN SAMPLE MEAN AND HISTORICAL DATA
    means= Returns.mean()                                                      #Calculate means for each asset's historical returns 
    diff_means= Returns.subtract(means)                                         #Calculate difference between historical return means and the historical returns

        #stage4: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values                                                   #Split historical returns from Dataframe index
    dates= diff_means.index                                                    #Split Dataframe index from historical returns

        #stage5: BUILD FORMULA
    md = []                                                                    #Define Mahalanobis Distance as md and create empty array for iteration
    for i in range(len(values)):
        md.append((np.dot(np.dot(np.transpose(values[i]),return_inverse),values[i])))#Construct Mahalanobis Distance formula and iterate over empty md array
        
        #stage6: CONVERT LIST TYPE TO DATAFRAME TYPE
    md_array= np.array(md)                                                     #Translate md List type to md Numpy Array type in order to join values into a Dataframe
    MD_daily=pd.DataFrame(md_array,index=dates,columns=list('R'))              #Join Dataframe index and Numpy array back together
    #MD_monthly= MD_daily.resample('M')                                         #resample data by average either as daily, monthly, yearly(ect.) 
   
   #---------------------------------------------------------------------------
    return    MD_daily                                                         #Return Malanobis Distance resampled returns, Malanobis Distance daily returns,  Turbulent returns and non-Turbulent returns
   #---------------------------------------------------------------------------


def MahalanobisDist_Turbulent_Returns(MD_returns, Returns):
    
    #Turbulent Returns
    turbulent= MD_returns[MD_returns>MD_returns.quantile(.75)[0]].dropna()
        #Day_with_Turbulent_returns
    returns=Returns
    returns['MD']=MD_returns
    Turbulent_Days=returns[returns['MD']>MD_returns.quantile(.75)[0]]
    Turbulent_Days= Turbulent_Days.drop('MD', 1)
    
    #Non_turbulent Returns
    non_turbulent=MD_returns[MD_returns<MD_returns.quantile(.75)[0]].dropna()
        #Day_with_non_Turbulent_returns
    non_Turbulent_Days=returns[returns['MD']<MD_returns.quantile(.75)[0]]
    non_Turbulent_Days= non_Turbulent_Days.drop('MD', 1)
    
    Returns=Returns.drop('MD',1)
    
   #---------------------------------------------------------------------------
    return turbulent, non_turbulent, Turbulent_Days,non_Turbulent_Days
   #---------------------------------------------------------------------------
    
   

def HistoricalTurbulenceIndexGraph( Mah_Days,  width, figsize, datesize):
    import matplotlib.pyplot as plt
    
    
    Monthly_Mah_Returns= Mah_Days.resample(datesize)   
    Monthly_Mah_Turbulent_Returns= Monthly_Mah_Returns[Monthly_Mah_Returns>Monthly_Mah_Returns.quantile(.75)[0]].dropna()    
    
    fig= plt.figure(1, figsize=figsize)
    plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
    plt.xlabel('Year')                                                          #label x axis Year
    plt.ylabel('Index')                                                         #label y axis Index
    plt.suptitle(['Historical Turbulence Index Calcualted from', datesize, 'Returns'],fontsize=12)   
    plt.bar(Monthly_Mah_Returns.index,Monthly_Mah_Returns.values, width,color='w', label='Quiet')#graph bar chart of Mahalanobis Distance
    plt.bar(Monthly_Mah_Turbulent_Returns.index,Monthly_Mah_Turbulent_Returns.values, width,color='k',alpha=0.8, label='Turbulent')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)      
    plt.show()
   
    #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------
    
    
#Journal Article: Kinlaw and Turkington - 2012 - Correlation Surprise
def Correlation_Surprise(Returns):
    
        #Stage1: IMPORT LIBRARIEs
    import pandas as pd                                                        #import pandas 
    import numpy as np                                                         #import numpy
     
         #Step1: CALCULATE TURBULENCE SCORE 
     
    #Stage 1: GENERATE TURBULENCE SCORE
    TS_daily= MahalanobisDist(Returns)                                               #calculate Turbulence Score from Mahalanobis Distance Function
    
             #Step2: CALCULATE MAGNITUDE SURPRISE   
    
        #Stage1: CALCULATE COVARIANCE MATRIX
    return_covariance= Returns.cov()                                           #Generate Covariance Matrix for hisotircal returns
    return_inverse= np.linalg.inv(return_covariance)                           #Generate Inverse Matrix for historical returns
    
        #stage2: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= Returns.mean()                                                      #Calculate historical returns means
    diff_means=Returns.subtract(means)                                         #Calculate difference between historical return means and the historical returns
    
        #stage3: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values                                                   #Split historical returns data from Dataframe
    dates= diff_means.index                                                    #Split Dataframe from historical returns
    
        #Stage4: Create Covariance and BLINDED MATRIX 
    inverse_diagonals=return_inverse.diagonal()                                #fetch only the matrix variances
    inverse_zeros=np.zeros(return_inverse.shape)                               #generate zeroed matrix with dynamic sizing properties 
    zeroed_matrix=np.fill_diagonal(inverse_zeros,inverse_diagonals)            #combine zeroed matrix and variances to form blinded matrix
    blinded_matrix=inverse_zeros                                               #define blinded matrix once the step above is completed
    
        #stage5: BUILD FORMULA
    ms = []                                                                    #Define Magnitude Surprise as ms                
    for i in range(len(values)):
        ms.append((np.dot(np.dot(np.transpose(values[i]),blinded_matrix),values[i])))       

        #stage6: CONVERT LIST Type TO DATAFRAME Type    
    ms_array= np.array(ms)                                                     #Translate ms List type to ts Numpy type
    Mag_Surprise_Daily=pd.DataFrame(ms_array,index=dates,columns=list('R'))               #Join Dataframe and Numpy array back together to calculate daily Magnitude Surprise Returns
    #MS=Mag_Surprise_Daily.resample('M')                                                   #create monthly returns for magnitude surprise
    
        
            #step3:CALCULATE CORRELATION SURPRISE
        #stage1: CALCULATE CORRELATION SURPRISE
    Corre_Surprise_Daily= TS_daily/(Mag_Surprise_Daily)   

                             # Calculate daily Correlation Surprise returns
    
    #Correlation_monthly_trail= Corre_Sur*Mag_Sur                                
    #resample_Correlation_monthly= Correlation_monthly_trail.resample('M',how=sum) 
    #MS_sum=Mag_Sur.resample('M',how=sum)                                       #Calculate monthly Magnitude Surprise returns 
    #Correlation_Surprise_monthly=resample_Correlation_monthly.divide(MS_sum)   #Calculate monthly Correlation Surprise retuns
    
    return  Corre_Surprise_Daily, Mag_Surprise_Daily               # Return Monthly Correlation Surprise Returns,  Monthly Magnitude Surprise returns, daily Correlation Surprise returns and daily magnitude surprise returns

   #---------------------------------------------------------------------------


def Corr_plot( Corr_sur, Mag_sur,  width, figsize, datesize):
    import matplotlib.pyplot as plt
    
    Corr_sur= Corr_sur.resample(datesize)
    Mag_sur= Mag_sur.resample(datesize)
    
    fig= plt.figure (figsize=figsize)
    fig.add_subplot(211)
    plt.xlabel('Year')                                                          #label x axis Year
    plt.ylabel('Index')                                                         #label y axis Index
    plt.suptitle(['Correlation Surprise and Magnitude Surprise', datesize, 'Returns'],fontsize=12)   
    plt.bar(Corr_sur.index,Corr_sur.values,color='w', width=width ,label= 'Correlation Surprise')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    
    fig.add_subplot(212)
    plt.xlabel('Year')                                                          #label x axis Year
    plt.ylabel('Index')                                                         #label y axis Index
    plt.bar(Mag_sur.index,Mag_sur.values,color='b', width=width ,label= 'Magnitude Surprise')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    

    plt.show()
    
   #---------------------------------------------------------------------------    
    return 
   #---------------------------------------------------------------------------
   #---------------------------------------------------------------------------



#Journal Article: Kritzman et al. - 2011 - Principal Components as a Measure of Systemic Risk
#http://www.mas.gov.sg/~/media/resource/legislation_guidelines/insurance/notices/GICS_Methodology.pdf
def Absorption_Ratio(Returns, halflife):
    
    #problem with Absorption ratio is that it needs non-log return data. Once this is obtained it should take the exponential 250 day returns. After the log returns should be taken and then the 500day trailing window    
    
        #stage0: IMPORT LIBRARIES    
    import pandas as pd                                                        #import pandas    
    import numpy as np                                                         #import numpys  
    import math as mth                                                         #import math
    from sklearn.decomposition import PCA

        #stage1: GATHER DAILY TRAIL LENGTH
    
    time_series_of_500days=len(Returns)-int(500/12)                              #collect data that is outside of initial 500day window
    
        #stage2: GENERATE ABSORPTION RATIO DATA
    plotting_data=[]                                                           #create list titled plot data
    for i in range(time_series_of_500days):
        
                #stage1: CALCULATE EXPONENTIAL WEIGHTING
        window= Returns[i:i+int(500/12)]                                  #create 500 day trailing window      
        #centred_data= returns_500day.subtract(returns_500day.mean())       #Center Data
        
        pca = PCA(n_components= int(round(Returns.shape[1]*0.2)), whiten=False).fit(window)
        Eigenvalues= pca.explained_variance_       
        
                    #stage6: CALCULATE ABSORPTION RATIO DATA
        variance_of_ith_eigenvector=Eigenvalues.sum()

        #variance_of_ith_eigenvector= np.var(Eigenvectors,axis=1).sum()
        #variance_of_ith_eigenvector= ev_vectors.diagonal()#fetch variance of ith eigenvector
        variance_of_jth_asset= window.var().sum()                        #fetch variance of jth asset
    
            #stage7: CONSTRUCT ABSORPTION RATIO FORMULA     
        numerator= variance_of_ith_eigenvector                                 #calculate the sum to n of variance of ith eigenvector
        denominator= variance_of_jth_asset                                     #calculate the sum to n of variance of jth asset
               
        Absorption_Ratio= numerator/denominator                                #calculate Absorption ratio
    
            #stage8: Append Data
        plotting_data.append(Absorption_Ratio)                                 #Append Absorption Ratio iterations into plotting_data list
        
    
         #stage9: Plot Data
    plot_array= np.array(plotting_data)                                        #convert plotting_data into array
    dates= Returns[int(500/12):time_series_of_500days+int(500/12)].index                  #gather dates index over 500 day window iterations
    Absorption_Ratio_daily=pd.DataFrame(plot_array,index=dates,columns=list('R'))#merge dates and Absorption ratio returns
    Absorption_Ratio_daily= pd.ewma(Absorption_Ratio_daily, halflife=halflife)
    #Absorption_Ratio=Absorption_Ratio_daily.resample('M', how=None)#group daily data into monthly data
    
    return  Absorption_Ratio_daily #, Eigenvectors                                                  #print Absorption Ratio

def Absorption_Ratio_VS_MSCI_Graph(MSCI, AR_returns):
    
    import matplotlib.pyplot as plt    
    
    fig=plt.figure(figsize=(10,5))
    
    ax1= fig.add_subplot(2,1,1, axisbg='white')
    plt.suptitle('Absorption Ratio vs US Stock Prices')   
    plt.xticks(rotation=50)
    plt.xlabel('Year')#label x axis Year
    ax1.set_ylabel('MSCI USA Price', color='b')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,MSCI.max()[0]*1.10))
    ax1.plot(MSCI.index[500:3152],MSCI.values[500:3152])
    
    
    ax2= ax1.twinx()
    plt.ylabel('Index')#label y axis Index
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,1.2))
    ax2.plot(AR_returns.index,AR_returns.values, 'g')
    ax2.set_ylabel('Absorption Ratio Index', color='g')

    plt.show()
    fig.savefig('Absorption Ratio_vs_US_Stock_Prices.png')
    
    return 


def Absorption_Ratio_Standardised_Shift(AR_Returns):    
    
    import pandas as pd
           
    AR_15DAY= pd.ewma(AR_Returns, span=15)
    AR_Yearly= pd.ewma(AR_Returns, span=253)
    AR_Variance= AR_Yearly.std()
    
    delta_AR= (AR_15DAY-AR_Yearly)/AR_Variance
    
   
    return delta_AR

def Absorption_Ratio_and_Drawdowns(delta_AR):    #how to measure all drawdowns
    prevmaxi = 0
    prevmini = 0
    maxi = 0

    for i in range(len(delta_AR))[1:]:
        if delta_AR['R'][i] >= delta_AR['R'][maxi]:
            maxi = i
        else:
      # You can only determine the largest drawdown on a downward price!
          if (delta_AR['R'][maxi] - delta_AR['R'][i]) > (delta_AR['R'][prevmaxi] - delta_AR['R'][prevmini]):
              prevmaxi = maxi
              prevmini = i
    return (delta_AR['R'][prevmaxi], delta_AR['R'][prevmini])


def plot_AR(AR, figsize, yaxis):
    
    import matplotlib.pyplot as plt
    
    
    plt.figure( figsize=(figsize))    
    plt.suptitle(['Absorption Ratio Index from',' Daily Returns'],fontsize=12) 
    plt.xticks(rotation=50)
    plt.xlabel('Year')
    plt.ylabel('Index')
    
    y1, y2 = yaxis
    plt.ylim([y1, y2])    
    
    #x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,0.5,1))
    AR= AR
    x=AR.index
    y=AR.values
    #plt.plot(x,y, linewidth=2.5, color='k')
    plt.bar(x,y, width=0.2,color='w', label='Quiet')
    plt.grid()
    #cannot seem to find out how to colour this?
    

    plt.show()

    return 


def plot_AR_ALL(US, UK, JPN, halflife):
        
    import matplotlib.pyplot as plt
    
    US_input= Absorption_Ratio(Returns= US, halflife=halflife)
    UK_input =Absorption_Ratio(Returns= UK, halflife=halflife)
    JPN_input =Absorption_Ratio(Returns= JPN, halflife=halflife)  
    
    plt.figure(figsize=(10,3))
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.5,1))
    plt.xlabel('Year')
    plt.ylabel('Absorption Ratio')
    plt.plot(US_input.index,US_input.values, label="US", linewidth=2, color = '0.2')
    plt.plot(UK_input.index, UK_input.values, label="UK", linewidth=3, linestyle='--', color = '0.1')
    plt.plot(JPN_input.index, JPN_input.values, label="JPN", linewidth=4, linestyle='-.', color = '0.05')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
    plt.grid()
    
    
    
    plt.show()
    
    return
#-------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------
def Probit(Input_Returns, vix,recession_data):
    import numpy as np
    import pandas as pd
            #Build Probit Dataframe
    Intial_window_Input=Input_Returns
    df=pd.DataFrame(index=MahalanobisDist(Returns=Intial_window_Input).index)  #will need to consider pushing this forward 500days due to 500AR window
    df['MD']=MahalanobisDist(Returns=Intial_window_Input[(41):])  
    Mag_Corr= Correlation_Surprise(Returns=Intial_window_Input[(41):])
    df['Mag_Corr']= Mag_Corr[1]/Mag_Corr[0]
    df['AR']=Absorption_Ratio(Returns= Intial_window_Input, halflife=int(500/12))
    #df['VIX']=vix.values
    df=df[int(41):] # Due to Absorption Ratio requiring 500day window
    df['Binary']=recession_data
                #A value of one is a recession period and a value of zero is an expandsion period
    #-----------------------------
    
    #Run Probit
    endog = df[['Binary']]      # Dependent
    exog = df[['MD','Mag_Corr','AR']]
               #'VIX']]  #Independent
  
    const = pd.Series(np.ones(exog.shape[0]), index=endog.index)
    const.name = 'Const'
    exog = pd.DataFrame([const, exog.MD, exog.Mag_Corr, exog.AR]).T
                         #exog.VIX]).T
    # Estimate the model
    import statsmodels.api as sm
    mod = sm.Probit(endog, exog)
    fit = mod.fit(disp=0)
    params = fit.params
    
    return params, df
#-----------------------------







"""
def AR_systemic_importance(AR):
    
    import numpy as np 
    
    Absorption_Ratio= AR[0]
    Top_20_eigenvectors= np.transpose(AR[1])
    
   
    #Calculate Centrality
        #Absolute value of the exposure of the ith asset within the jth eigenvector
    absolute_value_of_the_exposure_of_the_ith_asset_within_jth_eigenvector= []
    for i in range(len(np.transpose(Top_20_eigenvectors))):
        Relative_weights=[]
        for j in range(len(Top_20_eigenvectors)):
            jth_eigen_vector= Top_20_eigenvectors[j]
            weight= jth_eigen_vector[i]/(jth_eigen_vector.sum())
            Relative_weights.append(weight)
        
        
        
        
        
        
        absolute_value_of_the_exposure_of_the_ith_asset_within_jth_eigenvector.append(np.abs(np.sum(Relative_weights)))
            
   
    

    return absolute_value_of_the_exposure_of_the_ith_asset_within_jth_eigenvector

def Exhbit_8(delta_AR):
    
    
    return   
"""


    
"""
def Exhbit_9(Treasury_bonds, MSCIUS_PRICES):
    
    T_returns= logreturns(Returns=Treasury_bonds)
    MSCI_returns= logreturns(Returns=MSCIUS_PRICES)
    T_returns['MSCI']= MSCI_returns

    
    Returns=[]
    for i in range(1, len(T_returns)):
        Portfolio= T_returns[0:i]*[0.5,0.5]
        AR_Ratio= Absorption_Ratio(Returns= Portfolio)
        delta_AR=Absorption_Ratio_Standardised_Shift(AR_Returns=AR_Ratio)
        
        #it is     @it is not  #it is the same
        if delta_AR['R'][i]<delta_AR[0:i].quantile(.68)[0] and delta_AR['R'][i]>delta_AR[0:i].quantile(.32)[0]:
            Returns.append(Portfolio[i:i+1])
        
        elif delta_AR['R'][i]>delta_AR[0:i].quantile(.68)[0]:
            Portfolio= T_returns[0:i]*[0,1]
            AR_Ratio= Absorption_Ratio(Returns= Portfolio)
            delta_AR= Absorption_Ratio_Standardised_Shift(AR_Returns=AR_Ratio)
            Returns.append(Portfolio[i])
        
        elif delta_AR['R'][i]<delta_AR[0:i].quantile(.32)[0]:
            Portfolio= T_returns[0:i]*[1,0]
            AR_Ratio= Absorption_Ratio(Returns= Portfolio)
            delta_AR= Absorption_Ratio_Standardised_Shift(AR_Returns=AR_Ratio)
            Returns.append(Portfolio[i])
        
        

            
            
            
            
            
            
            
            
    
    #days with 
    AR_greater_one_std = delta_AR[delta_AR>delta_AR.quantile(.68)].dropna()
    
    
    #days with AR<-1o
    AR_less_one_std= delta_AR[delta_AR<delta_AR.quantile(.32)].dropna()
    
    
       
    return AR_greater_one_std, AR_less_one_std
    
    


#Plotting Systemic Risk Measures
def print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES):
    
   import matplotlib.pyplot as plt
    
   #1 MahalanobisDistances
   #1 MahalanobisDistance
   plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
   plt.xlabel('Year')                                                          #label x axis Year
   plt.ylabel('Index')                                                         #label y axis Index
   plt.suptitle('Mahalanobis Distance Index')                                  #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
   plt.bar(systemicRiskMeasure[0][0].index,systemicRiskMeasure[0][0].values, width=20,color='w', label='Quiet')#graph bar chart of Mahalanobis Distance
   plt.bar(systemicRiskMeasure[0][2].index,systemicRiskMeasure[0][2].values, width=20,color='k',alpha=0.8, label='Turbulent')
   plt.legend()
   plt.show()
 
   
   
   #2Correlation Surprise
#   Correlation_Surprise=systemicRiskMeasure[1][0]                              #gather Correlation surprise array
#   Magnitude_Surprise= systemicRiskMeasure[1][1]                               #gather turbulence score array
   
        #Magnitude Suprise   
  # plt.xticks(rotation=50)                                                    #rotate x axis labels 50 degrees
  # plt.xlabel('Year')                                                         #label x axis Year
  # plt.ylabel('Index')                                                        #label y axis Index
  # plt.suptitle('Magnitude Surprise Index')                                   #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
  # plt.bar(Magnitude_Surprise.index,Magnitude_Surprise.values, width=20)      #graph bar chart of Mahalanobis Distance
  # plt.show()
   
       #Correlation_Surprise
   #need to find weighted averaged return
#   plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
#   plt.xlabel('Year')                                                          #label x axis Year
#   plt.ylabel('Index')                                                         #label y axis Index
#   plt.suptitle('Correlation Surprise Index')                                  #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
#   plt.bar(Correlation_Surprise.index,Correlation_Surprise.values, width=2)     #graph bar chart of Mahalanobis Distance
#   plt.show()
   
   
   
   #3Absorption Ratio
   
#   fig=plt.figure()
    
#   ax1= fig.add_subplot(2,1,1, axisbg='white')
#   plt.suptitle('Absorption Ratio vs US Stock Prices')   
#   plt.xticks(rotation=50)
#   plt.xlabel('Year')#label x axis Year
#   ax1.set_ylabel('MSCI USA Price', color='b')
#   x1,x2,y1,y2 = plt.axis()
#   plt.axis((x1,x2,0,1600))
#   ax1.plot(MSCIUS_PRICES.index[500:3152],MSCIUS_PRICES.values[500:3152])

    
#   ax2= ax1.twinx()
   #plt.ylabel('Index')#label y axis Index
#   x1,x2,y1,y2 = plt.axis()
#   plt.axis((x1,x2,0,2))
#   ax2.plot(systemicRiskMeasure[2].index,systemicRiskMeasure[2].values, 'g')
#   ax2.set_ylabel('Absorption Ratio Index', color='g')

#   plt.show()
   
   
   
   

   
   
   
   
   
   
   
   
   #plt.xticks(rotation=50)  #rotate x axis labels 50 degrees
   #plt.xlabel('Year')#label x axis Year
   #plt.ylabel('Index')#label y axis Index
   #plt.suptitle('Absorption Ratio Index Calculated from Monthly Retuns of Yahoo Finance World Indices')#label title of graph Absorption Ratio
   #plt.plot(systemicRiskMeasure[2].index,systemicRiskMeasure[2].values)#graph line chart of Absorption Ratio
   #plt.show()
"""
 
 
 
 
 
 
 
 
"""
#stage2: CALCULATE COVARIANCE MATRIX
return_covariance= centred_data.cov()                                  #Generate Covariance Matrix over 500 day window
 
#stage3: CALCULATE EIGENVECTORS AND EIGENVALUES
ev_values,ev_vector= np.linalg.eig(return_covariance)                  #generate eigenvalues and vectors over 500 day window 
  
#Stage4: SORT EIGENVECTORS RESPECTIVE TO THEIR EIGENVALUES
ev_values_sort_high_to_low = ev_values.argsort()[::-1]                         
ev_values_sort=ev_values[ev_values_sort_high_to_low]                   #sort eigenvalues from highest to lowest
ev_vectors_sorted= ev_vector[:,ev_values_sort_high_to_low]             #sort eigenvectors corresponding to sorted eigenvalues
        
#Stage5: COLLECT 1/5 OF EIGENVALUES
shape= ev_vectors_sorted.shape[0]                                      #collect shape of ev_vector matrix
round_down_shape= mth.floor(shape*0.2)
#round_down_shape= mth.floor(shape*0.2) #round shape to lowest integer
ev_vectors= ev_vectors_sorted[:,0:round_down_shape]                    #collect 1/5th the number of assets in sample
"""

 
