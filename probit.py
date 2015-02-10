""" Probit Model
http://statsmodels.sourceforge.net/devel/examples/generated/example_discrete.html"""

# how to fit model http://statsmodels.sourceforge.net/devel/gettingstarted.html

import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

#Load data from Spector and Mazzeo (1980). Examples follow Greene's Econometric Analysis Ch. 21 (5th Edition).
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
spector_data.exog[:5, :]
spector_data.endog[:5]


#Probit Model
probit_mod = sm.Probit(spector_data.endog, spector_data.exog)
probit_res = probit_mod.fit()
print probit_res.params
probit_margeff = probit_res.get_margeff()
print probit_margeff.summary()

