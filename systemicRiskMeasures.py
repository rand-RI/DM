
def logreturns(Adjusted_Close_Prices):    #GENERATED LOGARITHMIC RETURNS
    
    import numpy as np    
        
    returns = np.log(Adjusted_Close_Prices/Adjusted_Close_Prices.shift(1)).dropna()  #Generate log returns
    resampled_data=returns.resample('d').dropna()                                    #Choose if Daily, Monthly, Yearly(ect) dataframe is required
    
    return   resampled_data                                                          #Return Log returns


                ##Systemic Risk Measures##

#Journal Article: Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
def MahalanobisDist(returns):#define MahalanobisDistance function
  
        #stage1: IMPORT LIBRARIES
    import pandas as pd                                                              #import pandas    
    import numpy as np                                                               #import numpy
    
        #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov()                                                 #Generate covariance matrix for historical returns
    return_inverse= np.linalg.inv(return_covariance)                                 #Generate inverse covariance matrix for historical returns

        #stage3: CALCULATE THE DIFFERENCE BETWEEN SAMPLE MEAN AND HISTORICAL DATA
    means= returns.mean()                                                            #Calculate means for each asset's historical returns 
    diff_means= returns.subtract(means)                                              #Calculate difference between historical return means and the historical returns

        #stage4: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values                                                         #Split historical returns from Dataframe index
    dates= diff_means.index                                                          #Split Dataframe index from historical returns

        #stage5: BUILD FORMULA
    md = []                                                                          #Define Mahalanobis Distance as md and create empty array for iteration
    for i in range(len(values)):
        md.append((np.dot(np.dot(np.transpose(values[i]),return_inverse),values[i])))#Construct Mahalanobis Distance formula and iterate over empty md array
        
        #stage6: CONVERT LIST TYPE TO DATAFRAME TYPE
    md_array= np.array(md)                                                           #Translate md List type to md Numpy Array type in order to join values into a Dataframe
    Mal_Dist=pd.DataFrame(md_array,index=dates,columns=list('R'))                    #Join Dataframe index and Numpy array back together
    MD= Mal_Dist.resample('M')                                                       #resample data by average either as daily, monthly, yearly(ect.) 
    
        #stage7: COLLECT TOP 75% PERCENTILE(Turbulent returns) AND BOTTOM 75% PERCENTILE(non-Turbulent returns) 
    turbulent= MD.loc[MD["R"]>float(MD.quantile(.75).as_matrix())]                   #Calculate top 75% percentile of Malanobis Distance returns (Turbulent returns)
    nonturbulent= MD.loc[MD["R"]<=float(MD.quantile(.75).as_matrix())]               #Calculate bottom 75% percentile of Malanobis Distance (non-Turbulent returns)
    
    return    MD, Mal_Dist, turbulent, nonturbulent                                   #Return Malanobis Distance resampled returns, Malanobis Distance daily returns,  Turbulent returns and non-Turbulent returns
  


def MahalanobisDist_Table1(returns): #have used returns and not returns_Figure5 as there is a singlur matrix error with linalg.inv(returns_Figure5.cov()) need to fix

        #stage 1: IMPORT LIBRARIES 
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #need to add an interating for_loop here for each column of the dataframe 
       
       #stage 2: IMPORT MALANOBIS DISTANCE RETURNS
    turbulence_score= MahalanobisDist(returns)[0]                                   #Collect Malanobis Distance resampled returns
    
       #stage 3: NORMALISE MALANOBIS DISTANCE RETURNS
    Normalised_Data=pd.DataFrame(index=turbulence_score.index)                      #Create open Dataframe with identical index as Malanobis Distance resampled returns 
    
    for i in range(len(turbulence_score.columns)):                                  #Iterate over range of total number of columns in Malanobis Distance resampled returns dataframe                       
        n=turbulence_score.columns[i]                                               #Iterate over Malanobis Distance resampled returns column titles
        m=turbulence_score[n]                                                       #Iterate over columns
    
        A=m.max()                                                                   #Calculate maximum value for each column
        B=m.min()                                                                   #Calculate minimum value for each column
        a,b=[0,1]                                                                   #Set Normalise Range with a=0 and b=1
        
        normalised_data=[]                                                          #Create Normalised Data empty array
        for i in range(len(turbulence_score)):                                      #Iterate over range of Malanobis Distance resampled returns index
            x= m[i]                                                                 #Let x equal the iterated row of previously itereated column
            normailse_value=(x-B)/(A-B)                                             #Calculate normalised return
            #normailse_value=(a+(x-A)*(b-a))/(B-A)
            normalised_data.append(normailse_value)                                 #Append normalised return with empty array
    
        Normalised_Data[n]= normalised_data #normalised data between 0-1            #Append normalised returns empty array to DataFrame
    
        #stage 4: GENERATE TABLE 1 DATA
    top_5=[]                                                                        #Create emply array of Top 5 days
    top_10=[]                                                                       #Create emply array of Top 10 days
    top_20=[]                                                                       #Create emply array of Top 20 days
    index=[]                                                                        #Create emply array of index
    
    for i in range(len(Normalised_Data)):                                           #Iterate over Normalised Data index
    
        Top_75_Percentile=Normalised_Data.loc[Normalised_Data["R"]>float(Normalised_Data.quantile(.75))] #Calculate Turbulent Days of Normalised returns
        Top_10_Percentile_of_75=float(Top_75_Percentile.quantile(.1))               #10% percentile of turbulent days / 10th Percentile Threshold

        if (Normalised_Data['R'][i]>Top_10_Percentile_of_75):                       #Determine all Normalised Data that is above the 10th Percentile Threshold
            x=Normalised_Data['R'][i+1:i+6].mean()                                  #Calcualte mean of 5 days after 
            y=Normalised_Data['R'][i+1:i+11].mean()                                 #Calcualte mean of 10 days after 
            z=Normalised_Data['R'][i+1:i+21].mean()                                 #Calcualte mean of 20 days after 
            zz=Normalised_Data.index[i]                                             #Calculate most turbulent days index 

            top_5.append(x)                                                         #Append mean of 5 days after to top_5 empty array
            top_10.append(y)                                                        #Append mean of 10 days after to top_10 empty array
            top_20.append(z)                                                        #Append mean of 20 days after to top_20 empty array
            index.append(zz)                                                        #Append index of most turbulent days to index empty array
    
    Table_1=pd.DataFrame(index=index)                                               # Create Table 1 DataFrame over most turbulent days index
    Table_1['5 Day']=top_5                                                          #Append top_5 array to Dataframe
    Table_1['10 Day']=top_10                                                        #Append top_10 array to Dataframe
    Table_1['20 Day']=top_20                                                        #Append top_20 array to Dataframe
    
    Table_1.sum()                                                                   #Calculate the sum of each Column in Table_1 Dataframe
    
        #stage5 : Plot Table 1    
    Table_1.plot(kind='bar', title='Mahalanobis Distance Table 1')                  #Plot bar graph of Table 1             
    plt.show()                                                                      # Show plot
                     
    return Table_1.dropna(), Top_75_Percentile                                      #Return Table 1,  return Top_75 Percentile of Normalised Data
    #need to find out how to calculate the percentile ranks
    
def MahalanobisDist_Table2(returns): #again need to  change returns due to singular matrix error in returns_figure5
     
    
    turbulent_returns=pd.DataFrame()
    turbulent_period= srm.MahalanobisDist_Table1(returns)[1]
    for i in range(len(turbulent_period)): 
        x=turbulent_period.index[i]
        
        
        
        for j in range(len(returns)):
            if x==returns.index[j]:
                y=returns[j:j+1]
                turbulent_returns= turbulent_returns.append(y)    
    
    
    
    #Conservative Portfolio
    WeightsC=[.2286, .1659, .4995, .0385, .0675, .0]
    Expected_returnC= (returns.sum() * WeightsC).sum()
    Full_sample_riskC= np.sqrt(np.diagonal((returns*WeightsC).cov()).sum())
    turbulent_riskC= np.sqrt(np.diagonal((turbulent_returns*WeightsC).cov()).sum())


    #Moderate Portfolio
    WeightsM=[.3523, .2422, .3281, .0259, .0516, .0]
    Expected_returnM= (returns.sum() * WeightsM).sum()
    Full_sample_riskM= np.sqrt(np.diagonal((returns*WeightsM).cov()).sum())
    turbulent_riskC= np.sqrt(np.diagonal((turbulent_returns*Weightsm).cov()).sum())

    #Aggressive Portfolio
    WeightsA=[.4815, .3219, .1489, .0128, .0349, .0]
    Expected_returnA= (returns.sum() * WeightsA).sum()
    Full_sample_riskA= np.sqrt(np.diagonal((returns*WeightsA).cov()).sum())
    turbulent_riskC= np.sqrt(np.diagonal((turbulent_returns*Weightsa).cov()).sum())
    
    
    #CREATE TABLE 
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
    turbulent_returns=pd.DataFrame()
    turbulent_period= srm.MahalanobisDist_Table1(returns)[1]
    for i in range(len(turbulent_period)): 
        x=turbulent_period.index[i]
        
        
        
        for j in range(len(returns)):
            if x==returns.index[j]:
                y=returns[j:j+1]
                turbulent_returns= turbulent_returns.append(y)
    
    #fix the name of C, M and A as it is repeated frmo above
    WeightsC=[.2286, .1659, .4995, .0385, .0675, .0]
    Expected_meanC= (turbulent_returns.mean() * WeightsC).mean()
    Full_sample_riskC= np.sqrt(np.diagonal((turbulent_returns*WeightsC).cov()).sum())
    VaRC= -Expected_meanC + 2.575*Full_sample_riskC

    WeightsM=[.3523, .2422, .3281, .0259, .0516, .0]   
    Expected_meanM= (turbulent_returns.mean() * WeightsC).mean()
    Full_sample_riskM= np.sqrt(np.diagonal((turbulent_returns*WeightsM).cov()).sum())
    VaRM= -Expected_meanM + 2.575*Full_sample_riskM
    
    WeightsA=[.4815, .3219, .1489, .0128, .0349, .0]
    Expected_meanA= (turbulent_returns.mean() * WeightsA).mean()
    Full_sample_riskA= np.sqrt(np.diagonal((turbulent_returns*WeightsA).cov()).sum())
    VaRA= -Expected_meanA + 2.575*Full_sample_riskA
         
     #need to calculate the Maximum loss and maximum drawdown
                 
    return     
    
    
def MahalanobisDist_Table4(returns): 
    
    Weights=[0.1,0.6,0.3]
    market_portfolio= returns[['TIP','^GSPC','^TYX']] * Weights
    
    #what are equilbirum returns?

        
    return     
    
def MahalanobisDist_Table5(returns): 
    
    Weights=[0.1,0.6,0.3]
    market_portfolio= returns[['TIP','^GSPC','^TYX']] * Weights
    
    #what are equilbirum returns?

        
    return    

def MahalanobisDist_Table6(returns): 
    
            
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
    
    return  Correlation_Surprise_monthly, MS, Corre_Sur, Mag_Sur
    



def Correlation_Surprise_Table_Exhbit5(SRM_correlationsurprise):
    import pandas as pd
    import numpy as np
    
       
#Step 1:
    Top_20_Percentile= SRM_correlationsurprise[3].loc[SRM_correlationsurprise[3]["R"]>float(SRM_correlationsurprise[3].quantile(.75))] #turbulent days
    
#Step2    
    CorSurp_20_Percentile_Mag=pd.DataFrame() # create Correlation DataFrame for Top 20% Magnitude Surprise dates
    MagSur_20= Top_20_Percentile    # Top 20% Magnitude Surprise
    for i in range(len(MagSur_20)): 
        x=MagSur_20.index[i]
               
        for j in range(len(SRM_correlationsurprise[2])):
            if x==SRM_correlationsurprise[2].index[j]: #If Top 20% Magnitude Surprise Dates equals the Correlation Surprise Data
                y=SRM_correlationsurprise[2][j:j+1]    # Grab the index and value data for previous
                CorSurp_20_Percentile_Mag= CorSurp_20_Percentile_Mag.append(y) #append to Dataframe
    
    Corr_greater_1= CorSurp_20_Percentile_Mag.loc[CorSurp_20_Percentile_Mag["R"]>float(1)] #Dates with Correlation Surprise greater than 1
    Corr_less_1= CorSurp_20_Percentile_Mag.loc[CorSurp_20_Percentile_Mag["R"]<float(1)]     #Dates with Correlation Surprise less than 1
    
 #Step3
    #Average Mag
    Average_MagSur_20= Top_20_Percentile.mean()  #Generate Average Magnitude Surprise values for Top 20%
    
    #Average Mag Corr>1
    MagSur_20_Greater_1 =pd.DataFrame()   #Create Magnitude Surprise with Correlation greater than 1 DataFrame
    MagSur_20= Top_20_Percentile
    for i in range(len(MagSur_20)): 
        x=MagSur_20.index[i]
               
        for j in range(len(Corr_greater_1)):
            if x==Corr_greater_1.index[j]:   # If Magnitude Surprise index equals the Correlation Surprise >1 index 
                y=MagSur_20[i:i+1]           #Grab Magnitude Surprise value
                MagSur_20_Greater_1= MagSur_20_Greater_1.append(y)  #Append Mag & Corr>1
     
    Average_MagSur_20_Greater_1= MagSur_20_Greater_1.mean()  #Generate Mean
               
    #Average Mag Corr<1
    MagSur_20_Less_1 =pd.DataFrame()
    MagSur_20= Top_20_Percentile
    for i in range(len(MagSur_20)): 
        x=MagSur_20.index[i]
               
        for j in range(len(Corr_less_1)):  
            if x==Corr_less_1.index[j]:   # If Magnitude Surprise index equals the Correlation Surprise <1 index
                y=MagSur_20[j:j+1]        #Grab Magnitude Surprise value
                MagSur_20_Less_1= MagSur_20_Less_1.append(y)  #Append Mag & Corr>1

    Average_MagSur_20_Less_1= MagSur_20_Less_1.mean()  #Generate Mean
    
    return Top_20_Percentile, Average_MagSur_20, Average_MagSur_20_Greater_1, Average_MagSur_20_Less_1, MagSur_20_Greater_1, MagSur_20_Less_1
    

def Correlation_Surprise_Table_Exhbit6(SRM_correlationsurprise, Correlation_Surprise_Exhibit_5):
    
    import pandas as pd
    import numpy as np 
    
    #Next day magnitude surprise
    Next_day_MagSur= pd.DataFrame()
    MagSur_20= Correlation_Surprise_Exhibit_5[0]
    for i in range(len(MagSur_20)): 
        x= MagSur_20.index[i]
               
        for j in range(len(SRM_correlationsurprise[3])):
            if x== SRM_correlationsurprise[3].index[j]:
                y= SRM_correlationsurprise[3][j+1:j+2]
                Next_day_MagSur= Next_day_MagSur.append(y)
    
    Average_Next_day_MagSur= Next_day_MagSur.mean()
    
    #Next day magnitude surprise with Correlation Surprise greater than 1
    Next_day_MagSur_Greater_1= pd.DataFrame()
    MagSur_20_Greater_1= Correlation_Surprise_Exhibit_5[4]
    for i in range(len(MagSur_20_Greater_1)): 
        x= MagSur_20_Greater_1.index[i]
               
        for j in range(len(SRM_correlationsurprise[3])):
            if x== SRM_correlationsurprise[3].index[j]:
                y= SRM_correlationsurprise[3][j+1:j+2]
                Next_day_MagSur_Greater_1= Next_day_MagSur_Greater_1.append(y)
    
    Average_Next_day_MagSur_Greater_1= Next_day_MagSur_Greater_1.mean()
    
    #Next day magnitude surprise with Correlation Surprise less than 1
    Next_day_MagSur_Less_1= pd.DataFrame()
    MagSur_20_Less_1= Correlation_Surprise_Exhibit_5[5]
    for i in range(len(MagSur_20_Less_1)): 
        x= MagSur_20_Less_1.index[i]
               
        for j in range(len(SRM_correlationsurprise[3])):
            if x== SRM_correlationsurprise[3].index[j]:
                y= SRM_correlationsurprise[3][j+1:j+2]
                Next_day_MagSur_Less_1= Next_day_MagSur_Less_1.append(y)
    
    Average_Next_day_MagSur_Less_1= Next_day_MagSur_Less_1.mean()
    
    
    return  Average_Next_day_MagSur,  Average_Next_day_MagSur_Greater_1,  Average_Next_day_MagSur_Less_1
        

def Correlation_Surprise_Table_Exhbit7(): 
    
            
    return    

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
    
    return  Absorption_Ratio #print Absorption Ratio
    
   
   
def Absorption_Ratio_Standardised_Shift(SRM_absorptionratio):    
    
    import pandas as pd
           
    AR_15DAY= pd.ewma(SRM_absorptionratio, span=15)
    AR_Yearly= pd.ewma(SRM_absorptionratio, span=253)
    AR_Variance= AR_Yearly.std()
    
    delta_AR= (AR_15DAY-AR_Yearly)/AR_Variance
    
   
    return delta_AR



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
   Correlation_Surprise=systemicRiskMeasure[1][0] #gather Correlation surprise array
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
   
