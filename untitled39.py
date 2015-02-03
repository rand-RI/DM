# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 17:42:18 2015

@author: Daniel Murray
"""



import pandas as pd
import systemicRiskMeasures as srm   

#-------------------------------------------------------------------------------
US_sectors= pd.load('USsectors')
US_sectors_returns= srm.logreturns(Returns=US_sectors)





#http://scipystats.blogspot.com.au/

import statsmodels.api as sm
import pandas as pd
import patsy

df=US_sectors_returns

f = 'y~duration+poutcome+month+contact+age+job';
y,x = patsy.dmatrices(f, df,return_type='dataframe')

"""
https://qizeresearch.wordpress.com/2013/12/03/quick-start-with-python-data-mining-using-pandas-patsy-and-statsmodels/
"""

"""
http://statsmodels.sourceforge.net/devel/examples/generated/example_discrete.html
"""