import pandas as pd
import quandl as Quandl, math
import numpy as np
from sklearn import preprocessing

api_k = "VKApkPmDCQfgmTvuE_xy"
df = Quandl.get('WIKI/GOOGL', api_key = api_k)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_chance'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT' , 'PCT_chance' , 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna('-99999', inplace = True)

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
print(df.head())
