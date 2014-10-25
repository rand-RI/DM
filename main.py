#stage1: IMPORT LIBRARY
import pandas.io.data as pdio      #import pandas.io.data library
import systemicRiskMeasures as srm #import Systemic Risk Measures library
import matplotlib.pyplot as plt
import pandas as pd
#stage2: IMPORT DATA WITH ANY NUMBER OF PORTFOLIOS 

          #Must arrange tickers in alphabetical order
symbols = ['EZA','FTSEMIB.MI','RTS.RS','^AORD','^ATX','^BFX','^BSESN','^BVSP','^FCHI','^FTSE','^GDAXI','^GSPTSE','^IXIC','^JKSE','^KS11','^MERV','^MXX','^N225','^SSEC'] # List all stock symbols to download in alphabetical order

#G20 MEMBERS#
#Argentina:^MERV is considered the most important index of Argentina's primary stock exchange "Buenos Aires Stock Exchange"
#Austrlaia: ^AORD is considered the oldest index of shares in Australia
#Brazil: ^BVSP is an index of 50 stocks that are traded on the Sao Paulo Stock Exchange in Brazil
#Canada:^GSPTSE is an index of the stock (equity) prices of the largest companies on the Toronto Stock Exchange in Canada
#China: ^SSEC represents all stocks that are traded on the Shanghai Stock Exchange
#France: ^FCHI is considered a benchmark French Stock Market index
#Germany: ^GDAXI is a  blue chip stock market index consisting of the 30 major German companies trading on the Frankfurt Stock Exchange.
#India:^BSESN SENSEX, is a free-float market-weighted stock market index of 30 well-established and financially sound companies listed on Bombay Stock Exchange.
#Indonesia:^JKSE  is an index of all stocks that trade on the Indonesia Stock Exchange
#Italy:FTSEMIB.MI  is the benchmark stock market index for the Borsa Italiana, the Italian national stock exchange
#Japan:^N225  is a stock market index for the Tokyo Stock Exchange (TSE). Currently, the Nikkei is the most widely quoted average of Japanese equities
#Republic of Korea: ^KS11 s the index of all common stocks traded on the Stock Market Division of the Korea Exchange
#Mexico:^MXX  The main benchmark stock index for the Mexican Stock Exchange is called the IPC
#Russia: RTS.RS
#Saudi Arabia: ...
#South Africa: EZA
#Turkey:
#United Kingdom: ^FTSE  is a share index of the 100 companies listed on the London Stock Exchange with the highest market capitalization
#United States of America:'^IXIC'  is a stock market index of the common stocks and similar securities listed on the NASDAQ stock market
#the European Union: 
#^ATX
#^BFX
#^FCHI
#^GDAXI
#^OMXSPI
#^SSMI

#stage3: DOWNLOAD DATA AND CALCULATE RETURN VALUES
Start_Date='1/1/2012'#MM,DD,YY
End_Date='12/11/2013'#MM,DD,YY
Historical_Prices = pdio.get_data_yahoo(symbols,start= Start_Date,end= End_Date) # Download data from YAHOO as a pandas Panel object
Adjusted_Close_Prices = Historical_Prices['Adj Close'].dropna()  # Scrape adjusted closing prices as pandas DataFrane object while also removing all Nan data

#daily returns:
returns = np.log(Adjusted_Close_Prices/Adjusted_Close_Prices.shift(1)).dropna()  # Continuously compounded returns while also removing top row of Nan data

#montly Returns:
returns_array=returns.values #n_array
d0=Adjusted_Close_Prices.index

x=Adjusted_Close_Prices.count()[0]
yyyymm=[]
for i in range(1,Adjusted_Close_Prices.count()[0]):
    a=d0.year[i]
    b="{0:02}".format(d0.month[i])
   
    yyyymm.append(str(int(str(a)+str(b))))
        
    
y=pd.DataFrame(returns_array,index=yyyymm,columns=[symbols])
monthly_re=y.groupby(y.index).sum()

#Add 30 dates ahead
Date=datetime.datetime.strptime(End_Date, "%m/%d/%Y")
Future_Date = Date + datetime.timedelta(days=31)
Future_Date_Read= Future_Date.strftime('%m/%d/%Y')

monthly_returns= pd.DataFrame(monthly_re.values,index=pd.date_range(Start_Date,Future_Date_Read, freq='M'),columns=[symbols])




#change monthly_returns to returns

#stage4: Import Systemic Risk Measures
SRM_mahalanobis= srm.MahalanobisDist(monthly_returns)       #define Mahalanobis Distance Formula
SRM_correlationsurprise= srm.Correlation_Surprise(monthly_returns)#define Correlation Surprise Score
SRM_absorptionratio= srm.Absorption_Ratio(returns)#define Absorption Ratio

systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio] # group systemic risk measures

#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
#    fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
#    fig.savefig("{}.jpg".format(sysRiskMeasure))

srm.print_systemic_Risk(systemicRiskMeasure)



monthly_returns.set_index(monthly_returns.index)
