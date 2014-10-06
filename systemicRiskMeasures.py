import pandas as pd
import numpy as np

def mahanalobis_distance(returns)

   

    #stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov() #Generate Covariance Matrix
    return_inverse= np.linalg.inv(returns.cov()) #Generate Inverse Matrix

    #stage3: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= returns.mean() #Calculate returns means
    diff_means= returns.subtract(means) #Calculate difference between return means and the scraped data

    #stage4: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values #Split scraped data from Dataframe
    dates= diff_means.index #Split Dataframe from scraped Data

    #stage5: BUILD FORMULA
    md = [] #Define Mahalanobis Distance as md
    for i in range(len(values)):
        md.append((np.dot(np.dot(np.transpose(values[i]),return_inverse),values[i])))

    #stage6: CONVERT LIST DATA TO DATAFRAME
    md_array= np.array(md) #Translate md List type to md Numpy type
    MD=pd.DataFrame(md_array,index=dates,columns=list('R')) #Join Dataframe and Numpy array back together

	return MD
	
def Correlation_Surprise(stock_data):

	#Stage2: CALCULATE COVARIANCE MATRIX
    return_covariance= returns.cov()             #Generate Covariance Matrix
    return_inverse= np.linalg.inv(returns.cov())           #Generate Inverse Matrix

    #stage3: CALCULATE THE DIFFERENCE BETWEEN THE MEAN AND HISTORICAL DATA FOR EACH INDEX
    means= returns.mean()                        #Calculate returns means
    diff_means= returns.subtract(means)          #Calculate difference between return means and the scraped data

    #stage4: SPLIT VALUES FROM INDEX DATES
    values=diff_means.values                        #Split scraped data from Dataframe
    dates= diff_means.index                        #Split Dataframe from scraped Data

    #stage5: BUILD FORMULA
    ts = []                                    #Define Turbulance Score as ts                
    for i in range(len(values)):
        ts.append((np.dot(np.dot(np.transpose(values[i]),return_inverse),values[i])))
    
    #stage6: CONVERT LIST DATA TO DATAFRAME
    ts_array= np.array(ts)                     #Translate md List type to ts Numpy type
    TS=pd.DataFrame(ts_array,index=dates,columns=list('R'))    #Join Dataframe and Numpy array back together




        #Step2: CALCULATE MAGNITUDE SURPRISE
    #Stage1: CREATE BLINDED MATRIX 
    inverse_diagonals= return_inverse.diagonal() #collect diagonals
    inverse_zeros=np.zeros(return_inverse.shape)   #find size of matrix so that prodecure is dynamic
    zeroed_matrix=np.fill_diagonal(inverse_zeros,inverse_diagonals)    #combind zeroed matrix and diagonals to form blinded matrix
    blinded_matrix=inverse_zeros                #define blinded matrix
    
    #stage2: BUILD FORMULA
    ms = []                                    #Define Magnitude Surprise as ms                
    for i in range(len(values)):
        ms.append((np.dot(np.dot(np.transpose(values[i]),blinded_matrix),values[i])))

    ms_array= np.array(ms)                     #Translate md List type to ts Numpy type
    MS=pd.DataFrame(ms_array,index=dates,columns=list('R')) 

   #Calculate Correlation Surprise
    Correlation_Surprise= TS.divide(MS)
	
	return Correlation_Surprise
	
def absorption_ratio:
	
#stage1: IMPORT DATA WITH ANY NUMBER OF PORTFOLIOS
symbols = ['^AORD','^HSI','^N225'] # List all stock symbols to download in alphabetical order
stock_data = get_data_yahoo(symbols,start='1/1/2005',end='1/1/2014') # Download data from YAHOO as a pandas Panel object

adj_close  = stock_data['Adj Close']         # Scrape adjusted closing prices as pandas DataFrane object
returns = log(adj_close/adj_close.shift(1))  # Continuously compounded returns

#stage2: subtract the mean
means= returns.mean()                       #Calculate return means
diff_means= returns.subtract(means)             

#stage3: CALCULATE COVARIANCE MATRIX
return_covariance= diff_means.cov()             #Generate Covariance Matrix

#stage4: CALCULATE EIGENVECTORS AND EIGENVALUES
ev_values,ev_vector= linalg.eig(return_covariance)         #generate eigenvalues and vectors 
ev_values_low_to_high=np.sort(ev_valuess)       #sort lowest to highest
ev_values_high_to_low= ev_values_low_to_high[::-1]          #sort highest to lowest	

return abs_ratio
	
   