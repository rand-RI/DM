import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices 

df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df[-5:]
df = df.dropna()
df[-5:]


# Y is an Nx1 column of data on lottery wagers per capita
#X is NX7 with an intercept, the Literacy & Weath variables , and 4 region binary variables
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')

