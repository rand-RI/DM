##Systemic Risk Measures##

#Journal Article: Kritzman and Li - 2010 - Skulls, Financial Turbulence, and Risk Management
def MahalanobisDist(returns):#define MahalanobisDistance function
    
    #stage1: IMPORT LIBRARIES
    import pandas as pd#import pandas    
    import numpy as np#import numpy
    
    #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix for historical returns
    return_inverse= np.linalg.inv(return_covariance) #Generate Inverse Matrix for historical returns

    #stage3: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
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
    MD=pd.DataFrame(md_array,index=dates,columns=list('R')) #Join Dataframe and Numpy array back together
    
    return MD #return Mahalanobis Distance data
    
#Journal Article: Kinlaw and Turkington - 2012 - Correlation Surprise
def Correlation_Surprise(returns):
    
    #Stage1: IMPORT LIBRARIES
    import pandas as pd#import pandas 
    import numpy as np #import numpy
     
         #Step1: CALCULATE TURBULENCE SCORE 
     
    #Stage 1: GENERATE TURBULENCE SCORE
    TS= MahalanobisDist(returns)#calculate Turbulence Score from Mahalanobis Distance Function
    
    
         #Step2: CALCULATE MAGNITUDE SURPRISE   
    
    #Stage1: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix for hisotircal returns
    return_inverse= np.linalg.inv(returns.cov()) #Generate Inverse Matrix for historical returns
    
    #stage2: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= returns.mean() #Calculate historical returns means
    diff_means= returns.subtract(means) #Calculate difference between historical return means and the historical returns
    
    #stage3: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values #Split historical returns data from Dataframe
    dates= diff_means.index #Split Dataframe from historical returns
    
    #Stage4: Create Covariance and BLINDED MATRIX 
    inverse_diagonals=return_inverse.diagonal() #fetch only the matrix variances
    inverse_zeros=np.zeros(return_inverse.shape) #generate zeroed matrix with dynamic sizing properties 
    zeroed_matrix=np.fill_diagonal(inverse_zeros,inverse_diagonals) #combind zeroed matrix and variances to form blinded matrix
    blinded_matrix=inverse_zeros #define blinded matrix
    
    #stage5: BUILD FORMULA
    ms = []                                    #Define Magnitude Surprise as ms                
    for i in range(len(values)):
        ms.append((np.dot(np.dot(np.transpose(values[i]),blinded_matrix),values[i])))       

    #stage6: CONVERT LIST Type TO DATAFRAME Type    
    ms_array= np.array(ms)  #Translate ms List type to ts Numpy type
    MS=pd.DataFrame(ms_array,index=dates,columns=list('R')) #Join Dataframe and Numpy array back together
    
    #stage7: STANDARDISE MS
    MS_Max= MS.max()#calculate maximum value of Turbulence Score data            
    MS_Standardised= MS.divide(MS_Max)#standardise data by dividing by maximum value
    
        #step3:CALCULATE CORRELATION SURPRISE
    #stage1: CALCULATE CORRELATION SURPRISE
    Correlation_Surprise= TS.divide(MS)
    
    #Stage 2: STANDARDISE CORRELATION SURPRISE
    Correlation_Surprise_Max= Correlation_Surprise.max()#calculate maximum value of Correlation Surprise    
    Correlation_Surprise_Standardised= Correlation_Surprise.divide(Correlation_Surprise_Max) #standardise data by dividing by maximum value
   
   
    return  Correlation_Surprise_Standardised, MS_Standardised #return standardised Correlation Surprise and Magnitude Surprise
    
#Journal Article: Kritzman et al. - 2011 - Principal Components as a Measure of Systemic Risk
def Absorption_Ratio(returns):
    
    #stage1: IMPORT LIBRARIES    
    import pandas as pd  #import pandas    
    import numpy as np #import numpys  
    
    #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix
    
    #stage3: CALCULATE EIGENVECTORS AND EIGENVALUES
    ev_values,ev_vector= np.linalg.eig(return_covariance) #generate eigenvalues and vectors 
    
    #Stage4: SORT EIGENVECTORS RESPECTIVE TO THEIR EIGENVALUES
    ev_values_sort_high_to_low = ev_values.argsort()[::-1]                         
    ev_values_sort=ev_values[ev_values_sort_high_to_low] #sort eigenvalues from highest to lowest
    ev_vectors_sorted= ev_vector[:,ev_values_sort_high_to_low] #sort eigenvectors corresponding to sorted eigenvalues
    
    
    #stage5: CALCULATE ABSORPTION RATIO DATA
    variance_of_ith_eigenvector= ev_vectors_sorted.diagonal()#fetch variance of ith eigenvector
    variance_of_jth_asset= np.array(returns).diagonal() #fetch variance of jth asset
    
    #stage6: CONSTRUCT ABSORPTION RATIO FORMULA     
    numerator= variance_of_ith_eigenvector.sum() #calculate the sum to n of variance of ith eigenvector
    denominator= variance_of_jth_asset.sum()#calculate the sum to n of variance of jth asset
    Absorption_Ratio= numerator/denominator#calculate Absorption ratio
    
    return Absorption_Ratio #return Absorption_Ratio

