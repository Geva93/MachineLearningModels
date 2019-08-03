import pandas as od
import quandl as Quandl

api_k = "VKApkPmDCQfgmTvuE_xy"
df = Quandl.get('WIKI/GOOGL', api_key = api_k)

print(df.head())
