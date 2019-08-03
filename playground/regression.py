import os
from os.path import join, dirname
from dotenv import load_dotenv
import pandas as pd
import quandl as Quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

style.use('ggplot')


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
secret_key = os.getenv('SECRET_KEY')

df = Quandl.get('WIKI/GOOGL', api_key = secret_key)

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True) #Fill Non available fields, in ML don't work with NaN values. Now it will be an outlier in the set.

forecast_out = int(math.ceil(0.01*len(df))) # round up to the nearst whole
df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'], 1))

x = preprocessing.scale(x) #normalize the data before fit
x = x[:-forecast_out]
x_lately = x[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
clf = LinearRegression(n_jobs = 1)
clf.fit(x_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(x_test, y_test)
#print(accuracy)
forecast_set = clf.predict(x_lately)
#print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
#plt.show()
