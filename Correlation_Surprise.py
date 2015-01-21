"""Correlation Surprise"""

def Correlation_Surprise(Returns):
    
        #Stage1: IMPORT LIBRARIEs
    import pandas as pd                                                        #import pandas 
    import numpy as np                                                         #import numpy
    import Mahalanobis_Distance as MD 
         #Step1: CALCULATE TURBULENCE SCORE 
     
    #Stage 1: GENERATE TURBULENCE SCORE
    TS= MD.MahalanobisDist(Returns)[1]                                            #calculate Turbulence Score from Mahalanobis Distance Function
    
    
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
    Mag_Sur=pd.DataFrame(ms_array,index=dates,columns=list('R'))               #Join Dataframe and Numpy array back together to calculate daily Magnitude Surprise Returns
    MS=Mag_Sur.resample('M')                                                   #create monthly returns for magnitude surprise
    
        
            #step3:CALCULATE CORRELATION SURPRISE
        #stage1: CALCULATE CORRELATION SURPRISE
    Corre_Sur= TS.divide(Mag_Sur)                                              # Calculate daily Correlation Surprise returns
    Correlation_monthly_trail= Corre_Sur*Mag_Sur                                
    resample_Correlation_monthly= Correlation_monthly_trail.resample('M',how=sum) 
    MS_sum=Mag_Sur.resample('M',how=sum)                                       #Calculate monthly Magnitude Surprise returns 
    Correlation_Surprise_monthly=resample_Correlation_monthly.divide(MS_sum)   #Calculate monthly Correlation Surprise retuns
    
    return  Correlation_Surprise_monthly, MS, Corre_Sur, Mag_Sur               # Return Monthly Correlation Surprise Returns,  Monthly Magnitude Surprise returns, daily Correlation Surprise returns and daily magnitude surprise returns
    


def Correlation_Surprise_Table_Exhbit5(Exhibit5_USEquities, Exhibit5_Euro_Equities, Exhibit5_Currency):
    import pandas as pd
        
    #stage1: Generate Correlation Surprise Returns for US Equities, European Equities and Currencies
    CS_Exhibit5_USEquities= Correlation_Surprise(Returns=Exhibit5_USEquities)  # CS returns for USEquities
    CS_Exhibit5_EuropeanEquities= Correlation_Surprise(Returns=Exhibit5_Euro_Equities)# CS returns for EuropeanEquities
    CS_Exhibit5_Currency= Correlation_Surprise(Returns=Exhibit5_Currency)      # CS returns for Currency
    Correlation_Measures=CS_Exhibit5_USEquities, CS_Exhibit5_EuropeanEquities, CS_Exhibit5_Currency #Group CS returns together labelled "Correlation_Measures"
    
    #Stage2: Calculate Exhibit 5 returns 
    Exhibit5_USEquities_returns=[]                                             #Create empty array  USEquities_return
    Exhibit5_EuropeanEquities_returns=[]                                       #Create empty array  EuroEquities_return
    Exhibit5_Currency=[]                                                       #Create empty array  Currency
    Exhibit5_returns= Exhibit5_USEquities_returns, Exhibit5_EuropeanEquities_returns, Exhibit5_Currency #Group Empty Arrays together

#Step1: Identify the 20% of days in the historical sample with the highest magnitude surprise scores    
    for l in range(len(Exhibit5_returns)):                                     #Begin by iterating over the index of Exhibit5_returns to calculate returns for US Equities, Euro Equities and Currency individually
        for k in range(len(Correlation_Measures)):                             #Iterate over the index of Correlation_Measures
            Top_20_Percentile= Correlation_Measures[k][3].loc[Correlation_Measures[k][3]["R"]>float(Correlation_Measures[k][3].quantile(.80))] #Calculate top 20% Magnitude Surprise returns for given returns
        
#Step2: Partition the sample from step 1 into two smaller subsamples: days with high correlation surprise and days with low correlation surprise    
            CorSurp_20_Percentile_Mag=pd.DataFrame()                           # create empty Correlation DataFrame for Top 20% Magnitude Surprise dates to be append to
            MagSur_20= Top_20_Percentile                                       # Defin Top 20% Magnitude Surprise as Magnitude Surprise 20%
            for i in range(len(MagSur_20)):                                    #Iterate over the index over MagSur_20
                x=MagSur_20.index[i]                                           #Let x equal dates of MagSur_20
                for j in range(len(Correlation_Measures[k][2])):               #Iterate over daily correlation surprise for US equities, Euro equities and currency indivudally
                    if x==Correlation_Measures[k][2].index[j]:                 #If Top 20% Magnitude Surprise Dates equals the Correlation Surprise Data
                        y=Correlation_Measures[k][2][j:j+1]                    #Grab the index and value data for that period
                        CorSurp_20_Percentile_Mag= CorSurp_20_Percentile_Mag.append(y) #append to Dataframe
    
            Corr_greater_1= CorSurp_20_Percentile_Mag.loc[CorSurp_20_Percentile_Mag["R"]>float(1)] #Dates with Correlation Surprise greater than 1
            Corr_less_1= CorSurp_20_Percentile_Mag.loc[CorSurp_20_Percentile_Mag["R"]<float(1)]#Dates with Correlation Surprise less than 1
    
#Step3: Measure, for the full smaple identified in step 1 and its two subsamples identified in step 2, the subsequent volatility and performance of relevant investments and strategies
    #Average Mag
            Average_MagSur_20= Top_20_Percentile.mean()                        #Generate Average Magnitude Surprise values for Top 20% over full sample
    
    #Average Mag Corr>1
            MagSur_20_Greater_1 =pd.DataFrame()                                #Create empty magnitude surprise dataframe for Magnitude Surprise returns with Correlation Surprise greater than 1
            MagSur_20= Top_20_Percentile                                       #Define top 20% of Magnitude Surprise
            for i in range(len(MagSur_20)):                                    #Iterate over index of Magnitude Surprise top 20%
                x=MagSur_20.index[i]    
                for j in range(len(Corr_greater_1)):                            
                    if x==Corr_greater_1.index[j]:                             #If Magnitude Surprise index equals the Correlation Surprise >1 index 
                        y=MagSur_20[i:i+1]                                     #Grab Magnitude Surprise value
                        MagSur_20_Greater_1= MagSur_20_Greater_1.append(y)     #Append Mag data given Corr>1
     
            Average_MagSur_20_Greater_1= MagSur_20_Greater_1.mean()            #Generate Mean 
               
    #Average Mag Corr<1
            MagSur_20_Less_1 =pd.DataFrame()
            MagSur_20= Top_20_Percentile
            for i in range(len(MagSur_20)): 
                x=MagSur_20.index[i]
                for j in range(len(Corr_less_1)):  
                    if x==Corr_less_1.index[j]:                                # If Magnitude Surprise index equals the Correlation Surprise <1 index
                        y=MagSur_20[j:j+1]                                     #Grab Magnitude Surprise value
                        MagSur_20_Less_1= MagSur_20_Less_1.append(y)           #Append Mag & Corr>1

            Average_MagSur_20_Less_1= MagSur_20_Less_1.mean()                  #Generate Mean
            
    #Append all data to each US Equities, and Euro Euities and Currencies indivudally       
            Exhibit5_returns[l].extend((Average_MagSur_20, Average_MagSur_20_Greater_1,Average_MagSur_20_Less_1))
    
    #Need to create table when importing different sets of data  
    #Rows= ['MS 20%','MS with CS>=1','MS 20% with CS<=1']
    #Table_5= pd.DataFrame(index= Rows)                                         #Create open Table_2
    
    #Table_5['US Equities']= Exhibit5_returns[0]                                #Append each Portfolio's Data to Table 2
    #Table_5['European Equities']= Exhibit5_returns[1]
    #Table_5['Currencies'] = Exhibit5_returns[2]
          
    
    #return  Table_5, Top_20_Percentile, Exhibit5_returns
    return  Top_20_Percentile, Exhibit5_returns
    
def Correlation_Surprise_Table_Exhbit6(SRM_correlationsurprise, Correlation_Surprise_Exhibit_5):
    
    import pandas as pd
     
    for l in range(len(Correlation_Surprise_Exhibit_5[1])):
        #Calculate Next day magnitude surprise
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
        MagSur_20_Greater_1= Correlation_Surprise_Exhibit_5[l][2]
        for i in range(len(MagSur_20_Greater_1)):
            x= MagSur_20_Greater_1.index[i]
            for j in range(len(SRM_correlationsurprise[3])):
                if x== SRM_correlationsurprise[3].index[j]:
                    y= SRM_correlationsurprise[3][j+1:j+2]
                    Next_day_MagSur_Greater_1= Next_day_MagSur_Greater_1.append(y)
    
        Average_Next_day_MagSur_Greater_1= Next_day_MagSur_Greater_1.mean()
    
    #Next day magnitude surprise with Correlation Surprise less than 1
        Next_day_MagSur_Less_1= pd.DataFrame()
        MagSur_20_Less_1= Correlation_Surprise_Exhibit_5[l][3]
        for i in range(len(MagSur_20_Less_1)): 
            x= MagSur_20_Less_1.index[i]
            for j in range(len(SRM_correlationsurprise[3])):
                if x== SRM_correlationsurprise[3].index[j]:
                    y= SRM_correlationsurprise[3][j+1:j+2]
                    Next_day_MagSur_Less_1= Next_day_MagSur_Less_1.append(y)
    
        Average_Next_day_MagSur_Less_1= Next_day_MagSur_Less_1.mean()
    
    #Tbale: when importing different sets of data  
    Rows= ['MS 20% with CS<=1' ,'MS 20%','MS with CS>=1']
    Table_6= pd.DataFrame(index= Rows)                                         #Create open Table_2
    
    Table_6['US Equities']= Exhibit5_returns[0]                                #Append each Portfolio's Data to Table 2
    Table_6['European Equities']= Exhibit5_returns[1]
    Table_6['Currencies'] = Exhibit5_returns[2]
    
    
    return  Average_Next_day_MagSur,  Average_Next_day_MagSur_Greater_1,  Average_Next_day_MagSur_Less_1
        

def Correlation_Surprise_Table_Exhbit7(): 
    
            
    return    