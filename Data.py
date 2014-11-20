import pandas.io.data as pdio      #import pandas.io.data library
import numpy as np

symbols = ['^AORD','^ATX','^BFX','^BSESN','^BVSP','^FCHI','^GDAXI','^GSPC','^GSPTSE','^HSI','^JKSE','^KLSE','^KS11','^MERV','^MXX','^N225','^SSEC','^STI','^TWII'] # List all stock symbols to download in alphabetical order
Start_Date='11/1/1980'#MM,DD,YY
End_Date='11/1/2014'#MM,DD,YY
frequency='d'

Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date) 
Historical_Prices.save(returns)
