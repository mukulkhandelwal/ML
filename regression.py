#Regression Intro


import pandas as pd
import  quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm 
from sklearn.linear_model import LinearRegression


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
y = np.array(df['label'])
print(y)
X = preprocessing.scale(X) #scale the values
print(X)
df.dropna(inplace=True)
y = np.array(df['label'])

#print(len(X),len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression() #LinearRegression(n_jobs=10) 10 times training data  # or clf = svm.SVR() for svm or  clf = svm.SVR(kernel = 'poly)
clf.fit(X_train, y_train)  #train
accuracy = clf.score(X_test, y_test)  #test

print(accuracy)


