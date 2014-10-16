#for the covariance should i calculat the means substract of returns first?

#Paper 1
def MahalanobisDist(returns):#define MahalanobisDistance function
    
    #stage1: IMPORT LIBRARIES
    import pandas as pd#import pandas    
    import numpy as np#import numpy
    
    #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix for returns
    return_inverse= np.linalg.inv(return_covariance) #Generate Inverse Matrix for returns

    #stage3: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= returns.mean()#Calculate returns means
    diff_means= returns.subtract(means) #Calculate difference between return means and the scraped data

    #stage4: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values #Split scraped data from Dataframe
    dates= diff_means.index #Split Dataframe from scraped Data

    #stage5: BUILD FORMULA
    md = [] #Define Mahalanobis Distance as md
    for i in range(len(values)):
        md.append((np.dot(np.dot(np.transpose(values[i]),return_inverse),values[i])))

    #stage6: CONVERT LIST Type TO DATAFRAME Type
    md_array= np.array(md) #Translate md List type to md Numpy type
    MD=pd.DataFrame(md_array,index=dates,columns=list('R')) #Join Dataframe and Numpy array back together
    
    return MD
    
#Paper 2
def Correlation_Surprise(returns):
    
    import pandas as pd#import pandas 
    import numpy as np #import numpy
     
         #Step1: CALCULATE TURBULENCE SCORE 
    TS= MahalanobisDist(returns)#calculate Turbulence Score from Mahalanobis Distance Function
    TS_Max= TS.max()
    TS_Standardised= TS.divide(TS_Max)
         #Step2: CALCULATE MAGNITUDE SURPRISE   
    #Stage1: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix for returns
    return_inverse= np.linalg.inv(returns.cov()) #Generate Inverse Matrix for returns
    
    #stage2: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= returns.mean() #Calculate returns means
    diff_means= returns.subtract(means) #Calculate difference between return means and the scraped data
    
    #stage3: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values #Split scraped data from Dataframe
    dates= diff_means.index #Split Dataframe from scraped Data
    
    #Stage4: Create Covariance and BLINDED MATRIX 
    inverse_diagonals=return_inverse.diagonal() #collect diagonals
    inverse_zeros=np.zeros(return_inverse.shape)   #find size of matrix so that procedure is dynamic
    zeroed_matrix=np.fill_diagonal(inverse_zeros,inverse_diagonals)    #combind zeroed matrix and diagonals to form blinded matrix
    blinded_matrix=inverse_zeros                #define blinded matrix
    
    #stage5: BUILD FORMULA
    ms = []                                    #Define Magnitude Surprise as ms                
    for i in range(len(values)):
        ms.append((np.dot(np.dot(np.transpose(values[i]),blinded_matrix),values[i])))

    ms_array= np.array(ms)                     #Translate md List type to ts Numpy type
    MS=pd.DataFrame(ms_array,index=dates,columns=list('R')) 
    
   #Calculate Correlation Surprise
    Correlation_Surprise= TS.divide(MS)
    Correlation_Surprise_Max= Correlation_Surprise.max()
    Correlation_Surprise_Standardised= Correlation_Surprise.divide(Correlation_Surprise_Max)
   
   
    return  Correlation_Surprise_Standardised, TS_Standardised


def Absorption_Ratio(returns):
    import pandas as pd    
    import numpy as np 
    
    #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov()             #Generate Covariance Matrix
    
    #stage3: CALCULATE EIGENVECTORS AND EIGENVALUES
    ev_values,ev_vector= np.linalg.eig(return_covariance)         #generate eigenvalues and vectors 
    ev_values_sort_high_to_low = ev_values.argsort()[::-1]                         
    ev_values_sort=ev_values[ev_values_sort_high_to_low]                               #sort eigenvalues from highest to lowest
    ev_vectors_sorted= ev_vector[:,ev_values_sort_high_to_low]                         #sort eigenvectors corresponding to sorted eigenvalues
    
    
    #stage5: CALCULATE ABSORPTION RATIO
    numerator= ev_vectors_sorted.diagonal()
    numerator_summed= numerator.sum()
    denominator= np.array(returns).diagonal()
    denominator_summed= denominator.sum()
    Absorption_Ratio= numerator_summed/denominator_summed
 
    return Absorption_Ratio

