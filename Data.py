import pandas.io.data as pdio      #import pandas.io.data library
import numpy as np

symbols = ['^GSPC','EFA','^TYX','TIP','VNQ','^DJC'] 
#['^GSPC','^NYA','^IXIC','^GSPC','^TYX','^DJC']
#['^AORD','^ATX','^BFX','^BSESN','^BVSP','^FCHI','^GDAXI','^GSPC','^GSPTSE','^HSI','^JKSE','^KLSE','^KS11','^MERV','^MXX','^N225','^SSEC','^STI','^TWII'] # List all stock symbols to download in alphabetical order
Start_Date='10/20/1973'#MM,DD,YY
End_Date='12/8/2014'#MM,DD,YY
frequency='d'

Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date) 
Historical_Prices.save('returns')


symbols = ['XWD.TO','^RUT','^GSPC','IWF','IWD','HDG'] 
Start_Date='1/1/1993'#MM,DD,YY
End_Date='12/8/2014'#MM,DD,YY
frequency='d'

Historical_Pricess = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date) 
Historical_Pricess_closed=Historical_Pricess['Adj Close'].dropna()
Figure5= pd.DataFrame(index=Historical_Pricess_closed.index)
Figure5['World Equties']=Historical_Pricess_closed['HDG'].values
Figure5['Small-Large']=Historical_Pricess_closed['^RUT'].values-Historical_Pricess_closed['^GSPC'].values
Figure5['Growth-Value']=Historical_Pricess_closed['IWF'].values-Historical_Pricess_closed['IWD'].values
Figure5['Hedge Funds']=Historical_Pricess_closed['HDG'].values
Figure5.save('returnsMD_Figure5')