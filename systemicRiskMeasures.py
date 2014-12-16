def logreturns(Adjusted_Close_Prices):

    import numpy as np    
    
    returns = np.log(Adjusted_Close_Prices/Adjusted_Close_Prices.shift(1)).dropna()
    resampled_data=returns.resample('d').dropna()
    
    return resampled_data


##Systemic Risk Measures##

#Journal Article: Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
def MahalanobisDist(returns):#define MahalanobisDistance function

#Figure 4:   
    #stage1: IMPORT LIBRARIES
    import pandas as pd#import pandas    
    import numpy as np#import numpy
    
    #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix for historical returns
    return_inverse= np.linalg.inv(return_covariance) #Generate Inverse Matrix for historical returns

    #stage3: CALCULATE THE DIFFERENCE BETWEEN THE SAMPLE MEAN AND HISTORICAL DATA
    means= returns.mean()#Calculate historical returns means for each asset
    diff_means= returns.subtract(means) #Calculate difference between historical return means and the historical returns

    #stage4: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values #Split historical returns from Dataframe
    dates= diff_means.index #Split Dataframe from historical returns

    #stage5: BUILD FORMULA
    md = [] #Define Mahalanobis Distance as md
    for i in range(len(values)):
        md.append((np.dot(np.dot(np.transpose(values[i]),return_inverse),values[i])))  #Construct Mahalanobis Distance formula
        
    #stage6: CONVERT LIST Type TO DATAFRAME Type
    md_array= np.array(md) #Translate md List type to md Numpy type
    Mal_Dist=pd.DataFrame(md_array,index=dates,columns=list('R')) #Join Dataframe and Numpy array back together
    MD= Mal_Dist.resample('M')#resample monthly data by average 
    
    #collect top 75% percentile 
    turbulent= MD.loc[MD["R"]>float(MD.quantile(.75).as_matrix())]
    #collect below 75% percentile
    nonturbulent= MD.loc[MD["R"]<=float(MD.quantile(.75).as_matrix())]
    
    return MD,Mal_Dist,turbulent,nonturbulent #return Mahalanobis Distance data
  

def MahalanobisDist_Table1(returns):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    turbulence_score= MahalanobisDist(returns)[0]
    
    Normalised_Data=pd.DataFrame(index=turbulence_score.index)
    for i in range(len(turbulence_score.columns)):
        n=turbulence_score.columns[i]
        m=turbulence_score[n]
    
        A=m.max()
        B=m.min()
        a=0
        b=1
    
        normalised_data=[]
        for i in range(len(turbulence_score)):
            x= m[i]
            normailse_value=(x-B)/(A-B)            
            #normailse_value=(a+(x-A)*(b-a))/(B-A)

            normalised_data.append(normailse_value)
    
        Normalised_Data[n]= normalised_data
    
    top_5=[]
    top_10=[]
    top_20=[]
    index=[]    
    
        
    for i in range(len(Normalised_Data)):
        Top_75_Percentile=Normalised_Data.loc[Normalised_Data["R"]>float(Normalised_Data.quantile(.75))] #turbulent days
        Top_10_Percentile_of_75=float(Top_75_Percentile.quantile(.1)) #10% percentile of turbulent days

        if (Normalised_Data['R'][i]>Top_10_Percentile_of_75):
            x=Normalised_Data['R'][i+1:i+6].mean()               #mean of 5 days after 
            y=Normalised_Data['R'][i+1:i+11].mean()              #mean of 10 days after 
            z=Normalised_Data['R'][i+1:i+21].mean()              #mean of 20 days after 
            zz=Normalised_Data.index[i]                          #most turbulent days index 

            top_5.append(x)
            top_10.append(y)
            top_20.append(z)
            index.append(zz)
    
    Table_1=pd.DataFrame(index=index)
    Table_1['5 Day']=top_5
    Table_1['10 Day']=top_10
    Table_1['20 Day']=top_20
    
    
    Table_1.plot(kind='bar', title='Mahalanobis Distance Table 1')
    plt.show()  
                     
    return Table_1.dropna(),Top_75_Percentile
    
def MahalanobisDist_Table2(returns): 
    
    #Conservative Portfolio
    WeightsC=[.2286, .1659, .4995, .0385, .0675, .0]
    Expected_returnC= (returns.sum() * WeightsC).sum()
    Full_sample_riskC= np.diagonal((returns*WeightsC).cov()).sum()



    #Moderate Portfolio
    WeightsM=[.3523, .2422, .3281, .0259, .0516, .0]
    Expected_returnM= (returns.sum() * WeightsM).sum()
    Full_sample_riskM= np.diagonal((returns*WeightsM).cov()).sum()


    #Aggressive Portfolio
    WeightsA=[.4815, .3219, .1489, .0128, .0349, .0]
    Expected_returnA= (returns.sum() * WeightsA).sum()
    Full_sample_riskA= np.diagonal((returns*WeightsA).cov()).sum()

    return     
    
     
def MahalanobisDist_Table3(returns): 
    
    #VaR for Full Sample, End of Horizon
    WeightsC=[.2286, .1659, .4995, .0385, .0675, .0]
    Expected_meanC= (returns.mean() * WeightsC).mean()
    Full_sample_riskC= np.sqrt(np.diagonal((returns*WeightsC).cov()).sum())
    VaRC= -Expected_meanC + 2.575*Full_sample_riskC

    WeightsM=[.3523, .2422, .3281, .0259, .0516, .0]   
    Expected_meanM= (returns.mean() * WeightsC).mean()
    Full_sample_riskM= np.sqrt(np.diagonal((returns*WeightsM).cov()).sum())
    VaRM= -Expected_meanM + 2.575*Full_sample_riskM
    
    WeightsA=[.4815, .3219, .1489, .0128, .0349, .0]
    Expected_meanA= (returns.mean() * WeightsA).mean()
    Full_sample_riskA= np.sqrt(np.diagonal((returns*WeightsA).cov()).sum())
    VaRA= -Expected_meanA + 2.575*Full_sample_riskA
    
    
    
    #Var for Turbulent periods
    returnss=[]
    
    # fix return values
    
    returns_turbulent= MahalanobisDist_Table1(returns)[1]
    Expected_meanC= (returns_turbulent.mean() * WeightsC).mean()
    Full_sample_riskC= np.sqrt(np.diagonal((*****returns}}}}}}}}}}*WeightsC).cov()).sum())
    VaRC= -Expected_meanC + 2.575*Full_sample_riskC




    returnss=[]
    returns_turbulent= srm.MahalanobisDist_Table1(returns)[1]
    for i in range(len(returns_turbulent)): 
        x=returns_turbulent.index[i]
        
        for j in range(len(returns)):
            if x==returns.index[j]:
                y=returns[i:i+1]
                returnss.append(y)
    
    
    
    x=pd.DataFrame()
    x.loc[returnss[0].index[0]]=returns[0]
   
    
    
    
    
    
     return     
    
    
    
#Journal Article: Kinlaw and Turkington - 2012 - Correlation Surprise
def Correlation_Surprise(returns):
    
    #Stage1: IMPORT LIBRARIES
    import pandas as pd#import pandas 
    import numpy as np #import numpy
     
         #Step1: CALCULATE TURBULENCE SCORE 
     
    #Stage 1: GENERATE TURBULENCE SCORE
    TS= MahalanobisDist(returns)[1]#calculate Turbulence Score from Mahalanobis Distance Function
    
    
         #Step2: CALCULATE MAGNITUDE SURPRISE   
    
    #Stage1: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix for hisotircal returns
    return_inverse= np.linalg.inv(return_covariance) #Generate Inverse Matrix for historical returns
    
    #stage2: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= returns.mean() #Calculate historical returns means
    diff_means= returns.subtract(means) #Calculate difference between historical return means and the historical returns
    
    #stage3: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values #Split historical returns data from Dataframe
    dates= diff_means.index #Split Dataframe from historical returns
    
    #Stage4: Create Covariance and BLINDED MATRIX 
    inverse_diagonals=return_inverse.diagonal() #fetch only the matrix variances
    inverse_zeros=np.zeros(return_inverse.shape) #generate zeroed matrix with dynamic sizing properties 
    zeroed_matrix=np.fill_diagonal(inverse_zeros,inverse_diagonals) #combine zeroed matrix and variances to form blinded matrix
    blinded_matrix=inverse_zeros #define blinded matrix
    
    #stage5: BUILD FORMULA
    ms = []                                    #Define Magnitude Surprise as ms                
    for i in range(len(values)):
        ms.append((np.dot(np.dot(np.transpose(values[i]),blinded_matrix),values[i])))       

    #stage6: CONVERT LIST Type TO DATAFRAME Type    
    ms_array= np.array(ms)  #Translate ms List type to ts Numpy type
    Mag_Sur=pd.DataFrame(ms_array,index=dates,columns=list('R')) #Join Dataframe and Numpy array back together
    MS=Mag_Sur.resample('M') #create monthly returns for magnitude surprise
    
        
        #step3:CALCULATE CORRELATION SURPRISE
    #stage1: CALCULATE CORRELATION SURPRISE
    Corre_Sur= TS.divide(Mag_Sur)
    
    Correlation_monthly_trail= Corre_Sur*Mag_Sur
    resample_Correlation_monthly= Correlation_monthly_trail.resample('M',how=sum)
    MS_sum=Mag_Sur.resample('M',how=sum)
    Correlation_Surprise_monthly=resample_Correlation_monthly.divide(MS_sum)
    
    return  Correlation_Surprise_monthly, MS
    



#Journal Article: Kritzman et al. - 2011 - Principal Components as a Measure of Systemic Risk
#http://www.mas.gov.sg/~/media/resource/legislation_guidelines/insurance/notices/GICS_Methodology.pdf
def Absorption_Ratio(FamaFrench49):
    
    #problem with Absorption ratio is that it needs non-log return data. Once this is obtained it should take the exponential 250 day returns. After the log returns should be taken and then the 500day trailing window    
    
    #stage1: IMPORT LIBRARIES    
    import pandas as pd  #import pandas    
    import numpy as np #import numpys  
    import math as mth #import math
    
    #stage1: GATHER DAILY TRAIL LENGTH
    time_series_of_500days=len(FamaFrench49)-500 #collect data that is outside of initial 500day window
    
    #stage2: GENERATE ABSORPTION RATIO DATA
    plotting_data=[]#create list titled plot data
        
    for i in range(time_series_of_500days):
        
              
        #stage1: CALCULATE EXPONENTIAL WEIGHTING
        returns_500day= FamaFrench49[i:i+500]#create 500 day trailing window        
        EWMA_returns=pd.ewma(returns_500day, halflife=250)
    
        #stage2: CALCULATE COVARIANCE MATRIX
        return_covariance= EWMA_returns.cov() #Generate Covariance Matrix over 500 day window
    
        #stage3: CALCULATE EIGENVECTORS AND EIGENVALUES
        ev_values,ev_vector= np.linalg.eig(return_covariance) #generate eigenvalues and vectors over 500 day window 
    
        #Stage4: SORT EIGENVECTORS RESPECTIVE TO THEIR EIGENVALUES
        ev_values_sort_high_to_low = ev_values.argsort()[::-1]                         
        ev_values_sort=ev_values[ev_values_sort_high_to_low] #sort eigenvalues from highest to lowest
        ev_vectors_sorted= ev_vector[:,ev_values_sort_high_to_low] #sort eigenvectors corresponding to sorted eigenvalues
    
        #Stage5: COLLECT 1/5 OF EIGENVALUES
        shape= ev_vectors_sorted.shape[0] #collect shape of ev_vector matrix
        round_down_shape= mth.floor(shape*0.2)
       #round_down_shape= mth.floor(shape*0.2) #round shape to lowest integer
        ev_vectors= ev_vectors_sorted[:,0:round_down_shape] #collect 1/5th the number of assets in sample
    
        #stage6: CALCULATE ABSORPTION RATIO DATA
        variance_of_ith_eigenvector= np.var(ev_vectors,axis=0).sum()
        #variance_of_ith_eigenvector= ev_vectors.diagonal()#fetch variance of ith eigenvector
        variance_of_jth_asset= EWMA_returns.var().sum() #fetch variance of jth asset
    
        #stage7: CONSTRUCT ABSORPTION RATIO FORMULA     
        numerator= variance_of_ith_eigenvector #calculate the sum to n of variance of ith eigenvector
        denominator= variance_of_jth_asset#calculate the sum to n of variance of jth asset
               
        Absorption_Ratio= numerator/denominator#calculate Absorption ratio
    
        #stage8: Append Data
        plotting_data.append(Absorption_Ratio) #Append Absorption Ratio iterations into plotting_data list
        
    
        #stage9: Plot Data
    plot_array= np.array(plotting_data)#convert plotting_data into array
    dates= FamaFrench49[500:time_series_of_500days+500].index#gather dates index over 500 day window iterations
    Absorption_Ratio_daily=pd.DataFrame(plot_array,index=dates,columns=list('R'))#merge dates and Absorption ratio returns
    Absorption_Ratio= Absorption_Ratio_daily
    #Absorption_Ratio=Absorption_Ratio_daily.resample('M', how=None)#group daily data into monthly data
    
        
    #exhibit 7
        
    AR_15DAY= pd.ewma(Absorption_Ratio, span=15)
    AR_Yearly= pd.ewma(Absorption_Ratio, span=253)
    AR_Variance= AR_Yearly.std()
    
    delta_AR= (AR_15DAY-AR_Yearly)/AR_Variance
    
    
    
    
    
    return  Absorption_Ratio #print Absorption Ratio
    
    #convert to monthly returns 

#Plotting Systemic Risk Measures
def print_systemic_Risk(systemicRiskMeasure,MSCIUS_PRICES):
    
   import matplotlib.pyplot as plt
    
   #1 MahalanobisDistance
   #1 MahalanobisDistance
   plt.xticks(rotation=50)  #rotate x axis labels 50 degrees
   plt.xlabel('Year')#label x axis Year
   plt.ylabel('Index')#label y axis Index
   plt.suptitle('Mahalanobis Distance Index')#label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
   plt.bar(systemicRiskMeasure[0][0].index,systemicRiskMeasure[0][0].values, width=20,color='w', label='Quiet')#graph bar chart of Mahalanobis Distance
   plt.bar(systemicRiskMeasure[0][2].index,systemicRiskMeasure[0][2].values, width=20,color='k',alpha=0.8, label='Turbulent')
   plt.legend()
   plt.show()
 
   
   
   #2Correlation Surprise
   Correlation_Surprise= systemicRiskMeasure[1][0] #gather Correlation surprise array
   Magnitude_Surprise= systemicRiskMeasure[1][1]#gather turbulence score array
   
        #Magnitude Suprise   
   plt.xticks(rotation=50)  #rotate x axis labels 50 degrees
   plt.xlabel('Year')#label x axis Year
   plt.ylabel('Index')#label y axis Index
   plt.suptitle('Magnitude Surprise Index')#label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
   plt.bar(Magnitude_Surprise.index,Magnitude_Surprise.values, width=20)#graph bar chart of Mahalanobis Distance
   plt.show()
   
       #Correlation_Surprise
   #need to find weighted averaged returns
   plt.xticks(rotation=50)  #rotate x axis labels 50 degrees
   plt.xlabel('Year')#label x axis Year
   plt.ylabel('Index')#label y axis Index
   plt.suptitle('Correlation Surprise Index')#label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
   plt.bar(Correlation_Surprise.index,Correlation_Surprise.values, width=20)#graph bar chart of Mahalanobis Distance
   plt.show()
   
   
   
   #3Absorption Ratio
   
   fig=plt.figure()
    
   ax1= fig.add_subplot(2,1,1, axisbg='white')
   plt.suptitle('Absorption Ratio vs US Stock Prices')   
   plt.xticks(rotation=50)
   plt.xlabel('Year')#label x axis Year
   ax1.set_ylabel('MSCI USA Price', color='b')
   x1,x2,y1,y2 = plt.axis()
   plt.axis((x1,x2,0,1600))
   ax1.plot(MSCIUS_PRICES.index[500:3152],MSCIUS_PRICES.values[500:3152])

    
   ax2= ax1.twinx()
   #plt.ylabel('Index')#label y axis Index
   x1,x2,y1,y2 = plt.axis()
   plt.axis((x1,x2,0,2))
   ax2.plot(systemicRiskMeasure[2].index,systemicRiskMeasure[2].values, 'g')
   ax2.set_ylabel('Absorption Ratio Index', color='g')

   plt.show()
   
   
   
   

   
   
   
   
   
   
   
   
   #plt.xticks(rotation=50)  #rotate x axis labels 50 degrees
   #plt.xlabel('Year')#label x axis Year
   #plt.ylabel('Index')#label y axis Index
   #plt.suptitle('Absorption Ratio Index Calculated from Monthly Retuns of Yahoo Finance World Indices')#label title of graph Absorption Ratio
   #plt.plot(systemicRiskMeasure[2].index,systemicRiskMeasure[2].values)#graph line chart of Absorption Ratio
   #plt.show()
   
