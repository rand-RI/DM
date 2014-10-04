def MahalanobisDist(stock_data):
    
    #import library
    import pandas as pd    
    import numpy as np
    
    
    #stage1: Extract required data
    adj_close = stock_data['Adj Close'] # Scrape adjusted closing prices as pandas DataFrane object
    returns = np.log(adj_close/adj_close.shift(1)) # Continuously compounded returns

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

    #stage7: plot
    import matplotlib.pyplot as plt
    MD.plot()               #Plot MD
    plt.ylabel('Index')
    plt.xlabel('Year')
    