def Correlation_Surprise(stock_data):
    #import Library
    import pandas as pd
    import numpy as np        
         
         
         #Step1: CALCULATE TURBULENCE SCORE    
    #Stage1: COLLECT DATA
    adj_close  = stock_data['Adj Close']         # Scrape adjusted closing prices as pandas DataFrane object
    returns = np.log(adj_close/adj_close.shift(1))  # Continuously compounded returns

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
   
   
    import matplotlib.pyplot as plt
    Correlation_Surprise.plot()                                   #Plot MD
    plt.ylabel('Index')                         
    plt.xlabel('Year')
    
    
