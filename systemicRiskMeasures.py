
def logreturns(Returns):    #GENERATED LOGARITHMIC RETURNS
    
    import numpy as np    
        
    returns = np.log(Returns/Returns.shift(1)).dropna()  #Generate log returns
    resampled_data=returns.resample('d').dropna()                              #Choose if Daily, Monthly, Yearly(ect) dataframe is required
    
    return   resampled_data                                                    #Return Log returns


                ##Systemic Risk Measures##

#Journal Article: Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
def MahalanobisDist(Returns):                                                  #define MahalanobisDistance function
  
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
    Mal_Dist=pd.DataFrame(md_array,index=dates,columns=list('R'))              #Join Dataframe index and Numpy array back together
    MD= Mal_Dist.resample('M')                                                 #resample data by average either as daily, monthly, yearly(ect.) 
    
        #stage7: COLLECT TOP 75% PERCENTILE(Turbulent returns) AND BOTTOM 75% PERCENTILE(non-Turbulent returns) 
    turbulent= MD.loc[MD["R"]>float(MD.quantile(.75).as_matrix())]             #Calculate top 75% percentile of Malanobis Distance returns (Turbulent returns)
    nonturbulent= MD.loc[MD["R"]<=float(MD.quantile(.75).as_matrix())]         #Calculate bottom 75% percentile of Malanobis Distance (non-Turbulent returns)
    
    return    MD, Mal_Dist, turbulent, nonturbulent                            #Return Malanobis Distance resampled returns, Malanobis Distance daily returns,  Turbulent returns and non-Turbulent returns
  


def MahalanobisDist_Table1(Returns): 

        #stage 1: IMPORT LIBRARIES 
    import pandas as pd
    import matplotlib.pyplot as plt
    Table_1_returns_Adjusted_Close= Returns['Adj Close'].dropna()              #Extract Adjusted Close returns from Table_1 returns

    
    """need to add an interating for_loop here for each column of the dataframe as mutiple dataframes will need to be passed through seperately(EG Global Assets, US assets....ect)"""
       
       #stage 2: IMPORT MAHALANOBIS DISTANCE RETURNS
    turbulence_score= MahalanobisDist(Returns=Table_1_returns_Adjusted_Close)[0]#Collect Malanobis Distance resampled returns
    
       #stage 3: NORMALISE MAHALANOBIS DISTANCE RETURNS
    Normalised_Data=pd.DataFrame(index=turbulence_score.index)                 #Create open Dataframe with identical index as Malanobis Distance resampled returns 
    for i in range(len(turbulence_score.columns)):                             #Iterate over range of total number of columns in Malanobis Distance resampled returns dataframe                       
        n=turbulence_score.columns[i]                                          #Iterate over Malanobis Distance resampled returns column titles
        m=turbulence_score[n]                                                  #Iterate over columns
        A=m.max()                                                              #Calculate maximum value for each column
        B=m.min()                                                              #Calculate minimum value for each column
        normalised_data=[]                                                     #Create Normalised Data empty array
        for j in range(len(turbulence_score)):                                 #Iterate over range of Malanobis Distance resampled returns index
            x= m[j]                                                            #Let x equal the iterated row of previously itereated column
            normailse_value=(x-B)/(A-B)                                        #Calculate normalised return
            normalised_data.append(normailse_value)                            #Append normalised return with empty array
        Normalised_Data[n]= normalised_data                                    #Append normalised returns empty array to DataFrame
    
        #stage 4: GENERATE TABLE 1 DATA
    Next_5=[]                                                                  #Create empty array of Next 5 days
    Next_10=[]                                                                 #Create empty array of Next 10 days
    Next_20=[]                                                                 #Create empty array of NExt 20 days
    index=[]                                                                   #Create empty array of index
    for i in range(len(Normalised_Data)):                                      #Iterate over Normalised Data index
        Top_75_Percentile=Normalised_Data.loc[Normalised_Data["R"]>float(Normalised_Data.quantile(.75))] #Calculate Turbulent Days of Normalised returns
        Top_10_Percentile_of_75=float(Top_75_Percentile.quantile(.1))          #10% percentile of turbulent days / 10th Percentile Threshold
        if (Normalised_Data['R'][i]>Top_10_Percentile_of_75):                  #Determine all Normalised Data that is above the 10th Percentile Threshold
            x=Normalised_Data['R'][i+1:i+6].mean()                             #Calcualte mean of 5 days after 
            y=Normalised_Data['R'][i+1:i+11].mean()                            #Calcualte mean of 10 days after 
            z=Normalised_Data['R'][i+1:i+21].mean()                            #Calcualte mean of 20 days after 
            zz=Normalised_Data.index[i]                                        #Collect most turbulent days index 
            Next_5.append(x)                                                   #Append mean of 5 days after to  Next_5 empty array
            Next_10.append(y)                                                  #Append mean of 10 days after to  Next_10 empty array
            Next_20.append(z)                                                  #Append mean of 20 days after to  Next_20 empty array
            index.append(zz)                                                   #Append index of most turbulent days to index empty array
    Table_1=pd.DataFrame(index=index)                                          #Create Table 1 DataFrame over most turbulent days index
    Table_1['Next 5 Days']= Next_5                                             #Append  Next_5 array to Dataframe
    Table_1['Next 10 Days']= Next_10                                           #Append  Next_10 array to Dataframe
    Table_1['Next 20 Days']= Next_20                                           #Append  Next_20 array to Dataframe
    
    Table_1_returns= Table_1.sum()                                             #Calculate the total average returns for each column in Table_1 Dataframe
    
        #stage5 : Plot Table 1    
    #Table_1.plot(kind='bar', title='Mahalanobis Distance Table 1')            #Plot bar graph of Table 1             
    #plt.show()     

                                                                 
    """need to find out how to calculate the percentile ranks"""    
                 
    return Table_1_returns, Top_75_Percentile                                  #Return Table 1,  return Top_75 Percentile of Normalised Data
    
    
def MahalanobisDist_Table2(Returns, Mahalanobis_Distance_Returns):             #need to source Table_2 returns.
    
    import pandas as pd
    import numpy as np    
    
        #stage 1: CREATE DATAFRAME OF RETURN VALUES FOR EVERY DATE THAT GENERATES A TOP 75% TURBULENCE SCORE 
    turbulent_returns=pd.DataFrame()                                           #Create open DataFrame
    turbulent_period= Mahalanobis_Distance_Returns[2]                          #import Top_75% of MahalanoBis Distance returns
    for i in range(len(turbulent_period)):                                     #Iterate over index of turbulent period
        x=turbulent_period.index[i]                                            #Let x equal iteration of turbulent period index
        for j in range(len(Returns)):                                          #Iterate over index of returns
            if x==Returns.index[j]:                                            #If returns index equals the normalised values index
                y=Returns[j:j+1]                                               #Collect the returns value row for every data that achieves a top 75% Malanobis Distance Turbulence score
                turbulent_returns= turbulent_returns.append(y)                 #Append
    
         #stage 2: CREATE PORTFOLIOS
    WeightsC=[.2286, .1659, .4995, .0385, .0675, .0]                           #Conservative Portfolio
    Expected_returnC= (Returns.sum() * WeightsC).sum()                         #Expected Returns
    Full_sample_riskC= np.sqrt(np.diagonal((Returns*WeightsC).cov()).sum())    #Full Sample Risk
    turbulent_riskC= np.sqrt(np.diagonal((turbulent_returns*WeightsC).cov()).sum()) #Turbulent Risk

    WeightsM=[.3523, .2422, .3281, .0259, .0516, .0]                           #Moderate Portfolio
    Expected_returnM= (Returns.sum() * WeightsM).sum()                         #Expected Returns
    Full_sample_riskM= np.sqrt(np.diagonal((Returns*WeightsM).cov()).sum())    #Full Sample Risk
    turbulent_riskM= np.sqrt(np.diagonal((turbulent_returns*WeightsM).cov()).sum())

    WeightsA=[.4815, .3219, .1489, .0128, .0349, .0]                           #Aggressive Portfolio
    Expected_returnA= (Returns.sum() * WeightsA).sum()                         #Expected Returns
    Full_sample_riskA= np.sqrt(np.diagonal((Returns*WeightsA).cov()).sum())    #Full Sample Risk
    turbulent_riskA= np.sqrt(np.diagonal((turbulent_returns*WeightsA).cov()).sum())
    
            #stage 3: Create Table
    Rows= ['US stocks','Non-US Stocks','US Bonds','Real Estate', 'Commodities', '', 'Expected Return', 'Full-Sample Risk', 'Turbulent Risk']
    Table_2= pd.DataFrame(index= Rows)                                         #Create open Table_2
    WeightsC.extend((Expected_returnC,Full_sample_riskC,turbulent_riskC))      #Add Expected Reuturns, Full Sample Risk and Tubulent Risk to Each Weights
    WeightsM.extend((Expected_returnM,Full_sample_riskM,turbulent_riskM))
    WeightsA.extend((Expected_returnA,Full_sample_riskA,turbulent_riskA))

    Table_2['Conservative Portfolio']= WeightsC                                #Append each Portfolio's Data to Table 2
    Table_2['Moderate portfolio']= WeightsM
    Table_2['Aggressive Portfolio'] = WeightsA
    
    
    
    return Table_2, turbulent_returns                                          #Returns Table_2 and Returns values that generate Tuburlent returns
    
     
def MahalanobisDist_Table3(Returns, Table_2): 
    
    from scipy.stats import norm                                               #This module contains a large number of probability distributions as well as a growing library of statistical functions.

    #Var for Full Sample, End of Horizon 
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma, on a portfolio
    of value P.
    """   
        #STAGE1: Create Empty Array's for each Type of Portfolio
    Conservative_Portfolio= []
    Moderate_Portfolio= []
    Aggressive_Portfolio = [] 
    
        #STAGE 2: Calculate VaR for Full Sample, End of Horizon
    for i in range(len(Table_2[0].columns)):
        n=Table_2[0].columns[i]                                                #Iterate over Table_2 column titles
        m=Table_2[0][n][0:6]                                                   #Collect weights for each portfolio from existing Table 2  
        P = 1e6   # 1,000,000 USD     #Need to find out how to do this?
        c = 0.01  # 1% confidence interval
        mu = (Returns*m.values).mean()                                         #Calculate return means given portfolio weights                
        sigma = (Returns*m.values).std()                                       #Calculate standard deviation given portfolio weights
        alpha = norm.ppf(1-c, mu, sigma)                                       #Calculate VaR alpha given type of portfolio 
        variance_Risk= P - P*(alpha + 1)                                       #Calcualte VaR
        if i==0:
            Conservative_Portfolio.append(variance_Risk)                       #Append VaR results for Conservative Portfolio to empty array 
        elif i==1:
            Moderate_Portfolio.append(variance_Risk)                           #Append VaR results for Moderate Portfolio to empty array 
        elif i==2:
            Aggressive_Portfolio.append(variance_Risk)                         #Append VaR results for AggressivePortfolio to empty array 
           
        #STAGE 3: Calculate VaR for Turbulent Sample , End of Horizon 
    for i in range(len(Table_2[0].columns)):
        n=Table_2[0].columns[i]                                                #Iterate over Malanobis Distance resampled returns column titles
        m=Table_2[0][n][0:6]
        P = 1e6   # 1,000,000 USD     #Need to find out how to do this?
        c = 0.01  # 1% confidence interval
        mu = (Table_2[1]*m.values).mean()                       
        sigma = (Table_2[1]*m.values).std()
        alpha = norm.ppf(1-c, mu, sigma) 
        variance_Risk= P - P*(alpha + 1)
        if i==0:
            Conservative_Portfolio.append(variance_Risk)                       #Append VaR results for Conservative Portfolio to empty array 
        elif i==1:
            Moderate_Portfolio.append(variance_Risk)                           #Append VaR results for Moderate Portfolio to empty array 
        elif i==2:
            Aggressive_Portfolio.append(variance_Risk)                         #Append VaR results for AggressivePortfolio to empty array 
            
    
    #what is the maximum loss and maximum drawdown 
 
                
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
def Correlation_Surprise(Returns):
    
        #Stage1: IMPORT LIBRARIEs
    import pandas as pd                                                        #import pandas 
    import numpy as np                                                         #import numpy
     
         #Step1: CALCULATE TURBULENCE SCORE 
     
    #Stage 1: GENERATE TURBULENCE SCORE
    TS= MahalanobisDist(Returns)[1]                                            #calculate Turbulence Score from Mahalanobis Distance Function
    
    
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
    


def Correlation_Surprise_Table_Exhbit5(Exhibit5_USEquities, Exhibit5_EuropeanEquities, Exhibit5_Currency):
    import pandas as pd
        
    #stage1: Generate Correlation Surprise Returns for US Equities, European Equities and Currencies
    CS_Exhibit5_USEquities= Correlation_Surprise(Returns=Exhibit5_USEquities)  # CS returns for USEquities
    CS_Exhibit5_EuropeanEquities= Correlation_Surprise(Returns=Exhibit5_EuropeanEquities)# CS returns for EuropeanEquities
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

#Journal Article: Kritzman et al. - 2011 - Principal Components as a Measure of Systemic Risk
#http://www.mas.gov.sg/~/media/resource/legislation_guidelines/insurance/notices/GICS_Methodology.pdf
def Absorption_Ratio(FamaFrench49):
    
    #problem with Absorption ratio is that it needs non-log return data. Once this is obtained it should take the exponential 250 day returns. After the log returns should be taken and then the 500day trailing window    
    
        #stage0: IMPORT LIBRARIES    
    import pandas as pd                                                        #import pandas    
    import numpy as np                                                         #import numpys  
    import math as mth                                                         #import math
    
        #stage1: GATHER DAILY TRAIL LENGTH
    time_series_of_500days=len(FamaFrench49)-500                               #collect data that is outside of initial 500day window
    
        #stage2: GENERATE ABSORPTION RATIO DATA
    plotting_data=[]                                                           #create list titled plot data
    for i in range(time_series_of_500days):
        
                #stage1: CALCULATE EXPONENTIAL WEIGHTING
        returns_500day= FamaFrench49[i:i+500]                                  #create 500 day trailing window        
        EWMA_returns=pd.ewma(returns_500day, halflife=250)
    
            #stage2: CALCULATE COVARIANCE MATRIX
        return_covariance= EWMA_returns.cov()                                  #Generate Covariance Matrix over 500 day window
    
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
    
            #stage6: CALCULATE ABSORPTION RATIO DATA
        variance_of_ith_eigenvector= np.var(ev_vectors,axis=0).sum()
        #variance_of_ith_eigenvector= ev_vectors.diagonal()#fetch variance of ith eigenvector
        variance_of_jth_asset= EWMA_returns.var().sum()                        #fetch variance of jth asset
    
            #stage7: CONSTRUCT ABSORPTION RATIO FORMULA     
        numerator= variance_of_ith_eigenvector                                 #calculate the sum to n of variance of ith eigenvector
        denominator= variance_of_jth_asset                                     #calculate the sum to n of variance of jth asset
               
        Absorption_Ratio= numerator/denominator                                #calculate Absorption ratio
    
            #stage8: Append Data
        plotting_data.append(Absorption_Ratio)                                 #Append Absorption Ratio iterations into plotting_data list
        
    
         #stage9: Plot Data
    plot_array= np.array(plotting_data)                                        #convert plotting_data into array
    dates= FamaFrench49[500:time_series_of_500days+500].index                  #gather dates index over 500 day window iterations
    Absorption_Ratio_daily=pd.DataFrame(plot_array,index=dates,columns=list('R'))#merge dates and Absorption ratio returns
    Absorption_Ratio= Absorption_Ratio_daily
    #Absorption_Ratio=Absorption_Ratio_daily.resample('M', how=None)#group daily data into monthly data
    
    return  Absorption_Ratio                                                   #print Absorption Ratio
    
   
   
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
   Correlation_Surprise=systemicRiskMeasure[1][0]                              #gather Correlation surprise array
   Magnitude_Surprise= systemicRiskMeasure[1][1]                               #gather turbulence score array
   
        #Magnitude Suprise   
  # plt.xticks(rotation=50)                                                    #rotate x axis labels 50 degrees
  # plt.xlabel('Year')                                                         #label x axis Year
  # plt.ylabel('Index')                                                        #label y axis Index
  # plt.suptitle('Magnitude Surprise Index')                                   #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
  # plt.bar(Magnitude_Surprise.index,Magnitude_Surprise.values, width=20)      #graph bar chart of Mahalanobis Distance
  # plt.show()
   
       #Correlation_Surprise
   #need to find weighted averaged return
   plt.xticks(rotation=50)                                                     #rotate x axis labels 50 degrees
   plt.xlabel('Year')                                                          #label x axis Year
   plt.ylabel('Index')                                                         #label y axis Index
   plt.suptitle('Correlation Surprise Index')                                  #label title of graph Historical Turbulence Index Calculated from Daily Retuns of G20 Countries
   plt.bar(Correlation_Surprise.index,Correlation_Surprise.values, width=2)     #graph bar chart of Mahalanobis Distance
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
   
