#Regression Intro

import pandas as pd
import  quandl
import math

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

forecast_out = int(math.ceil(0.01*len(df)))  


#labels
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())







