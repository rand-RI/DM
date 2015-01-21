
""" This Python scripts downloads all the required data for each Systemic Risk Paper
    1. Malanhobis Distance
    2. Correlation Surprise
    3. Absorption Ratio
    ....                                                                            """

                #STAGE1: Import Python libraries 

import pandas.io.data as pdio                                                  #Functions from pandas.io.data extract data from various Internet sources into a DataFrame. Currently the following sources are supported: Yahoo Finance, Google Finance, FRED, Kenneth French's library, World Bank and Google Analytics  
import numpy as np                                                             #NumPy is an extension to the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays
import pandas as pd                                                            #Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas.io.data as web
import systemicRiskMeasures as srm   

                #STAGE2: Download Required Data

        # MahalanoBis Distance
#Figure 4
#symbols = ['^GSPC','EFA','^TYX','TIP','VNQ','^DJC']                            #An example of the type of data that would be required for Mahalanobis Distance inputs
#Start_Date,End_Date='10/20/1973','12/8/2014'                                   #Date Range for required returns. In the format of MM/DD/YY
#Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date)# Download Historical Prices using symbols as a list of yahoo tickers over the defined start date-end date
#Historical_Prices.save('returns')  

#symbols = ['^TNX']                            #An example of the type of data that would be required for Mahalanobis Distance inputs
#Start_Date,End_Date='01/01/1998','1/29/2010'                                   #Date Range for required returns. In the format of MM/DD/YY
#Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date)# Download Historical Prices using symbols as a list of yahoo tickers over the defined start date-end date
#Historical_Prices.save('Treasury_Bonds_AR_Exhibit_9')                                            #Save Figure 4's data as 'returns' 

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
Global_Assets=pd.read_csv('Global_Assets.csv', index_col=0)
Global_Assets_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=Global_Assets)
Global_Assets_timeseries_dataframe.save('Table1_Global_Assets')

US_Assets=pd.read_csv('US_Assets.csv', index_col=0)
US_Assets_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=US_Assets)
US_Assets_timeseries_dataframe.save('Table1_US_Assets')

Currency=pd.read_csv('Currency.csv', index_col=0)
Currency_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=Currency)
Currency_timeseries_dataframe.save('Table1_Currency')

#Table 2
Table_2_Asset_Classes=pd.read_csv('MahDis_Table_2.csv', index_col=0)
Asset_Classes_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=Table_2_Asset_Classes)
Asset_Classes_timeseries_dataframe_log_returns=srm.logreturns(Returns=Asset_Classes_timeseries_dataframe) 
Asset_Classes_timeseries_dataframe_log_returns.save('Table2_Asset_Classes')

#Mean-variance optimisation
symbols = ['^GSPC','EFA','TIP','VNQ']                                                             #An example of the type of data that would be required for Figure 5 inputs 
Start_Date,End_Date='09/30/2004','12/8/2014'                                     #Date Range for required returns. In the format of MM/DD/YY 
Historical_Pricess = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date)#Download Historical Prices using symbols as a list of yahoo tickers over the defined start date-end date
Historical_Pricess_closed=Historical_Pricess['Adj Close'].dropna()  
Historical_Pricess_closed.save('S&P_500')    

        #Correlation Surprise
#Exhibit 5: US Equities,  European Equities, Currencies 
    #Equities 
US_Equities=pd.read_csv('CorrSur_USEquities.csv', index_col=0)
US_Equities_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=US_Equities)
US_Equities_timeseries_dataframe.save('Exhibit5_US_Equities')
    #European Equtiies 
Euro_Equities=pd.read_csv('CorrSur_EuropeanEquities.csv', index_col=0)
Euro_Equities_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=Euro_Equities)
Euro_Equities_timeseries_dataframe.save('Exhibit5_Euro_Equities')
    #Currency
Currency=pd.read_csv('CorrSur_Currency.csv', index_col=0)
Currency_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=Currency)
Currency_timeseries_dataframe.save('Exhibit5_Currency')



#tstart= '10/20/2000'                                                           #An example of the type of data required for Exhibit 5 Currency
#tend= '10/20/2014'
#Currency= web.DataReader(["EXUSEU","EXUSUK"], 'fred', tstart, tend) 
#Currency.save('CorrelationSurprise_Exhibit5_Currency')


#Images for Article
    #Mahalanobis Data
EM_Asia=pd.read_csv('EM_Asia_xIndia.csv', index_col=0)
EM_Asia_timeseries_dataframe=srm.CreateDataFrameWithTimeStampIndex(DataFrame=EM_Asia)
EM_Asia_timeseries_dataframe.save('EM_Asia_xIndia')




