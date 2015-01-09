
""" This Python scripts downloads all the required data for each Systemic Risk Paper
    1. Malanhobis Distance
    2. Correlation Surprise
    3. Absorption Ratio
    ....                                                                            """

                #STAGE1: Import Python libraries 

import pandas.io.data as pdio                                                  #Functions from pandas.io.data extract data from various Internet sources into a DataFrame. Currently the following sources are supported: Yahoo Finance, Google Finance, FRED, Kenneth French's library, World Bank and Google Analytics  
import numpy as np                                                             #NumPy is an extension to the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays
import pandas as pd                                                            #Pandas is a software library written for the Python programming language for data manipulation and analysis.

                #STAGE2: Download Required Data

        # MahalanoBis Distance
#Figure 4
symbols = ['^GSPC','EFA','^TYX','TIP','VNQ','^DJC']                            #An example of the type of data that would be required for Mahalanobis Distance inputs
Start_Date,End_Date='10/20/1973','12/8/2014'                                   #Date Range for required returns. In the format of MM/DD/YY
Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date)# Download Historical Prices using symbols as a list of yahoo tickers over the defined start date-end date
Historical_Prices.save('returns')                                              #Save Figure 4's data as 'returns' 

#Figure 5 MD
symbols = ['XWD.TO','^RUT','^GSPC','IWF','IWD','HDG']                          #An example of the type of data that would be required for Figure 5 inputs 
Start_Date,End_Date='1/1/1993','12/8/2014'                                     #Date Range for required returns. In the format of MM/DD/YY 
Historical_Pricess = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date)#Download Historical Prices using symbols as a list of yahoo tickers over the defined start date-end date
Historical_Pricess_closed=Historical_Pricess['Adj Close'].dropna()             # From Historical Prices extract the Adjusted CLose Returns 
Figure5= pd.DataFrame(index=Historical_Pricess_closed.index)                   #Create Figure 5 DataFrame with Historical_Prices dates as index.             
Figure5['World Equties']=Historical_Pricess_closed['HDG'].values               #Append World Equties to DataFrame 
Figure5['Small-Large']=Historical_Pricess_closed['^RUT'].values-Historical_Pricess_closed['^GSPC'].values#Append Small-Large to DataFrame
Figure5['Growth-Value']=Historical_Pricess_closed['IWF'].values-Historical_Pricess_closed['IWD'].values#Append Growth-Value to DataFrame
Figure5['Hedge Funds']=Historical_Pricess_closed['HDG'].values                 #Append Hedge Funds to DataFrame
Figure5.save('returnsMD_Figure5')                                              #save Figure 5's DataFrame as returns Mahalanobis Distance Figure 5

#Table 1
symbols = ['GMWAX','AAT','IYF']                                                #Import Table 1 Data for Malanobis Distance Paper
Start_Date,End_Date='1/1/1993','12/8/2014'                                     #MM,DD,YY
Historical_Pricess = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date)#Download Historical Prices using symbols as a list of yahoo tickers over the defined start date-end date
Historical_Pricess.save('Table_1')                                             #save Table 1 returns as Table_1 


        #Correlation Surprise
#Exhibit 5: US Equities,  European Equities, Currencies 
    #Equities 
symbols = ['^GSPC','^OEX']                                                     #An example of the type of data required for Exhibit 5 US equities
Start_Date,End_Date='10/20/2000','12/8/2014'#MM,DD,YY
Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date) 
Historical_Prices['Adj Close'].save('CorrelationSurprise_Exhibit5_USEquities')
    #European Equtiies 
symbols = ['^FTSE','^ATX']                                                     #An example of the type of data required for Exhibit 5 Euro equities
Start_Date,End_Date='10/20/2000','12/8/2014'#MM,DD,YY
Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date) 
Historical_Prices['Adj Close'].save('CorrelationSurprise_Exhibit5_EuropeanEquities')
    #Currency
tstart= '10/20/2000'                                                           #An example of the type of data required for Exhibit 5 Currency
tend= '10/20/2014'
Currency= web.DataReader(["EXUSEU","EXUSUK"], 'fred', tstart, tend) 
Currency.save('CorrelationSurprise_Exhibit5_Currency')






