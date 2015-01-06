#stage1: IMPORT LIBRARY
import pandas.io.data as pdio      #import pandas.io.data library
import systemicRiskMeasures as srm #import Systemic Risk Measures library
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt 
import numpy as np
import pickle as pk

#stage 1: Set Data
symbols = ['^AORD','^ATX','^BFX','^BSESN','^BVSP','^FCHI','^GDAXI','^GSPC','^GSPTSE','^HSI','^JKSE','^KLSE','^KS11','^MERV','^MXX','^N225','^SSEC','^STI','^TWII'] # List all stock symbols to download in alphabetical order
Start_Date='11/1/1980'#MM,DD,YY
End_Date='11/1/2014'#MM,DD,YY
frequency='d'

#Stage 2: Pulldata
returns= srm.pulldata(symbols, Start_Date,End_Date,frequency)[0]


#Stage 3: Import Systemic Risk Measures and Run Pulled Data

SRM_mahalanobis= srm.MahalanobisDist(returns)[0]#define Mahalanobis Distance Formula
SRM_correlationsurprise= srm.Correlation_Surprise(returns)#define Correlation Surprise Score
SRM_absorptionratio= srm.Absorption_Ratio(returns)#define Absorption Ratio

#output = pk.open('data.pk1','wb')
#pk.dump(data)

# Creating data frame for systemic risk measures


systemicRiskMeasure_df = pd.DataFrame(SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio)
systemicRiskMeasure_df.columns = ['Mahanalobis','CorrelationSurprise','AbsorptionRatio']





#systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio] # group systemic risk measures
#srm.print_systemic_Risk(systemicRiskMeasure)



#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
 #   fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
  #  fig.savefig("{}.jpg".format(sysRiskMeasure))