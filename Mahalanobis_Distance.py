"""
Mahalanobis Distance
"""
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
    




