from alpha_vantage.foreignexchange import ForeignExchange
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override() # <== that's all it takes :-)
import FundamentalAnalysis as fa
import nltk
import streamlit as st
import matplotlib.pyplot as plt, pandas as pd, numpy as np
import matplotlib
from PIL import Image
matplotlib.use('Agg')
from fbprophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
from alpha_vantage.foreignexchange import ForeignExchange
from matplotlib.pyplot import rc
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from dateutil.parser import parse
from scipy.stats import iqr
from datetime import timedelta, date
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import tweepy
import pandas as pd
from textblob import TextBlob
import preprocessor as p
#from preprocessor import api
import re

timeframe='15min'
ticker ='USDCHF'

def get_historical(ticker):
    cc = ForeignExchange(key='N6A6QT6IBFJOPJ70', output_format='pandas')
    data, meta_data = cc.get_currency_exchange_intraday(from_symbol=ticker[0:3], to_symbol=ticker[3:], interval=timeframe ,outputsize='full')
    df = pd.DataFrame(data)
    #df= pd.date_range(pd.to_datetime('today'), periods=10, freq='15min')
    df= df.rename(columns={'date': 'Date', '1. open': 'Open','2. high':'High','3. low': 'Low','4. close':'Close'})
    df.index.names = ['Date']
    #df['Close'].plot()
    #plt.tight_layout()
    #plt.title('15 mins EUR/USD')
    #df = df.to_csv(quote + ".csv")
    #plt.show()
    #print()
    #print("##############################################################################")
    #print("Today's price")
    #today_stock = df.iloc[1:2]
    #print(today_stock)
    #print("##############################################################################")
    #print()
    return df

def create_dataset(dataset, time_step=300):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

def reset_my_index(df):
  res = df[::-1].reset_index(drop=True)
  return(res)



from sklearn.preprocessing import MinMaxScaler
df = get_historical('EURUSD')
df = reset_my_index(df)['Close']

#Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

training_size = int(len(df1) * 0.7)

test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]


time_step =100 # time_pred(timeframe)
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

#print(X_train.shape), print(y_train.shape)
#print(X_test.shape), print(ytest.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(5, return_sequences=False,input_shape=(100, 1)))
#model.add(LSTM(50, return_sequences=True))
#model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=5, batch_size=64, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

import math
from sklearn.metrics import mean_squared_error

x= math.sqrt(mean_squared_error(y_train, train_predict))

x = math.sqrt(mean_squared_error(ytest, test_predict))


#fig = plt.figure(figsize=(10, 6), dpi=100)
look_back = 100 #time_pred(timeframe)
import numpy
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
#print(train_predict)
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
# plot baseline and predictions


plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.legend(loc='best')
plt.plot(testPredictPlot)
plt.grid(True)
plt.show()
plt.tight_layout()
#st.pyplot()


x_input = test_data[len(test_data) - 100:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()


lst_output = []
n_steps = 100
i = 0
while (i < 300):

    if (len(temp_input) > n_steps):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        # print("{} 15 mins input {}".format(i,x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} 15 mins output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1

# print(lst_output)

#day_new = np.arange(1, 101)
#day_pred = np.arange(101, 131)

df3 = df1.tolist()
df3.extend(lst_output)

predictions = scaler.inverse_transform(lst_output)
mid = len(predictions) / 2

if ((predictions[0] > predictions[int(mid)]) and (predictions[0] > predictions[-1])):
    out1= str("SELL @ %3f" % predictions[0])
    exit1 = str("TAKE PROFIT @ %3f" % predictions[299])
elif ((predictions[0] < predictions[int(mid)] and predictions[0] < predictions[-1])):
    out1 = str("BUY @ %3f" % predictions[0])
    exit1 = str("TAKE PROFIT @ %3f" % predictions[299])
else:
    out1= "HOLD"
    exit1 = 'HOLD'

x1 = len(df3) - 300
history = scaler.inverse_transform(df3[:-300])
prediction = scaler.inverse_transform(df3[int(x1):])


