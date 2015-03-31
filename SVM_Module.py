"SVM"


"""STAGE 1: 
IMPORT LIBRARY"""
#-------------------------
import pandas as pd
import systemicRiskMeasures as srm   
import matplotlib.pyplot as plt    
                                      #Import Systemic Risk Measures library
#----------------------------------------------------------------------------------------------------------------------------------


"""STAGE 2: 
IMPORT DATA"""
#--------------------------
US_sectors= pd.load('USsectors')
US_sectors_returns= srm.logreturns(Returns=US_sectors)

#-------------------------
#----------------------------------------------------------------------------------------------------------------------------------


"""STAGE 3: 
IMPORT SYSTEMIC RISK MEASURES AND RUN SIGNALS"""
#-------------------------
Input= US_sectors_returns.resample('M') #input monthly returns

"""Mahalanobis Distance"""
        #Input
MD_input=Input           #Change this value for data required
        #Run
SRM_mahalanobis= srm.MahalanobisDist(Returns=MD_input)   
SRM_mahalanobis_turbulent_nonturbulent_days= srm.MahalanobisDist_Turbulent_Returns(MD_returns= SRM_mahalanobis, Returns=MD_input)
                    #drop inputs
Input=Input.drop('MD',1)
MD_input= MD_input.drop('MD',1)
#-------------------------

"""Correlation Surprise"""
        #Input
Corr_Input= Input
        #Run
SRM_Correlation_Surprise=srm.Correlation_Surprise(Returns=Corr_Input)
Correlation_Surprise=SRM_Correlation_Surprise[1]/SRM_Correlation_Surprise[0]
#-------------------------

"""Absorption Ratio"""
        #Input
AR_input= Input
        #Run
SRM_absorptionratio= srm.Absorption_Ratio(Returns= AR_input, halflife=int(500/12))                        #define Absorption Ratio

#Group Data togerther
All_Measures= pd.DataFrame(index=Input.index)
All_Measures['MD']=SRM_mahalanobis
All_Measures['COR']=Correlation_Surprise
All_Measures['AR']=SRM_absorptionratio
All_Measures=All_Measures.dropna()
#-------------------------------

"""SVM"""
from sklearn import svm
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC()
clf.fit(X, Y) 
dec = clf.decision_function([[1]])
dec.shape[1] 

