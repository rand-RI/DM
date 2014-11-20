#stage1: IMPORT LIBRARY
import pandas.io.data as pdio      #import pandas.io.data library
import systemicRiskMeasures as srm #import Systemic Risk Measures library
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt 
import numpy as np

#Stage 1: Load Data
Historical_Prices= pd.load('returns')

#Stage 2: Pulldata
returns= srm.logreturns(Historical_Prices)

#Stage 3: Import Systemic Risk Measures and Run Pulled Data
SRM_mahalanobis= srm.MahalanobisDist(returns)[0]#define Mahalanobis Distance Formula
SRM_correlationsurprise= srm.Correlation_Surprise(returns)#define Correlation Surprise Score
SRM_absorptionratio= srm.Absorption_Ratio(returns)#define Absorption Ratio
systemicRiskMeasure= [SRM_mahalanobis,SRM_correlationsurprise,SRM_absorptionratio] # group systemic risk measures
srm.print_systemic_Risk(systemicRiskMeasure)



#sysRiskMeasure=0
#for sysRiskMeasure in systemicRiskMeasure:
 #   fig= srm.print_systemic_Risk(systemicRiskMeasure[sysRiskMeasure])
  #  fig.savefig("{}.jpg".format(sysRiskMeasure))