#Regression Intro
from datetime import datetime

import pandas as pd
import  quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = 'QM1RfC8UNPe6_AqSSde4'
quandl.ApiConfig.api_version = '2015-04-09'

df = quandl.get('WIKI/GOOGL')
#print(df)
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print(df)

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True) #nan data

forecast_out = int(math.ceil(0.01*len(df)))   #how many days in advance you want to print


#labels
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
#print(df.head())

X = np.array(df.drop(['label'],1)) #features except label
 #scale the values

X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
#X = X[:-forecast_out:]

#print(X)

df.dropna(inplace=True)
#y = np.array(df['label'])
y = np.array(df['label'])

#print(len(X),len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression() #LinearRegression(n_jobs=10) 10 times training data  # or clf = svm.SVR() for svm or  clf = svm.SVR(kernel = 'poly)
clf.fit(X_train, y_train)  #train
accuracy = clf.score(X_test, y_test)  #test

#print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy,forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1) ]+ [i]


df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()