import FundamentalAnalysis as fa
import nltk
import yfinance as yf
yf.pdr_override()
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
np.random.seed(4)
import tensorflow
tensorflow.random.set_seed(4)
import streamlit as st
import matplotlib.pyplot as plt, pandas as pd, numpy as np
import matplotlib
from PIL import Image
from callback import LossAndErrorPrintingCallback
matplotlib.use('Agg')
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras.layers.recurrent import GRU
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
import re


import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True


def set_pub():
    rc('font', weight='bold')    # bold fonts are easier to see
    rc('grid', c='0.5', ls='-', lw=0.5)
    rc('figure', figsize = (10,8))
    plt.style.use('bmh')
    rc('lines', linewidth=1.3, color='b')

@st.cache(suppress_st_warning=True)
def loadData(ticker, start, end):
     df_stockdata = pdr.get_data_yahoo(ticker, start= str(start), end = str(end))
     df_stockdata.index = pd.to_datetime(df_stockdata.index)
     return df_stockdata


def lstm_model_tech(lstm_input,dense_input,ohlcv_train,tech_ind_train,y_train):
        # the first branch operates on the first input
        x = LSTM(25, name='lstm_0')(lstm_input)
        x = Dropout(0.1, name='lstm_dropout_0')(x)
        lstm_branch = Model(inputs=lstm_input, outputs=x)

        # the second branch opreates on the second input
        y = Dense(25, name='tech_dense_0')(dense_input)
        y = Activation("relu", name='tech_relu_0')(y)
        y = Dropout(0.1, name='tech_dropout_0')(y)
        technical_indicators_branch = Model(inputs=dense_input, outputs=y)

        # combine the output of the two branches
        combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

        z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
        z = Dense(1, activation="linear", name='dense_out')(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
        adam = optimizers.Adam(lr=0.0005)
        model.compile(loss='mean_squared_error',optimizer=adam)
        model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=10, epochs=20, shuffle=True, validation_split=0.3)

        return model

def tech_ind():
    data = loadData(ticker,start,end)
    #data = data.drop('Datetime', axis=1)
    data = data.drop('Volume', axis=1)
    data = data.drop('Adj Close', axis=1)
    data = data.drop('High', axis=1)
    data = data.drop('Open', axis=1)
    data = data.drop('Low', axis=1)
    data = data.values

    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    history_points = 50
    import numpy as np
    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])


    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])


    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
    y_normaliser = preprocessing.MinMaxScaler()

    y_normaliser.fit(next_day_open_values)

    technical_indicators = []
    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 0])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][0]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    for his in ohlcv_histories_normalised:
        #Taking the SMA of the closing price to calcualte the MACF
        sma = np.mean(his[:, 0])
        macd = calc_ema(his, 12) - calc_ema(his, 26)

        technical_indicators.append(np.array([macd]))
        # technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array(technical_indicators)
    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)
    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]

    test_split = 0.7
    n = int(ohlcv_histories_normalised.shape[0] * test_split)

    ohlcv_train = ohlcv_histories_normalised[:n]
    tech_ind_train = technical_indicators_normalised[:n]
    y_train = next_day_open_values_normalised[:n]
    ohlcv_test = ohlcv_histories_normalised[n:]
    tech_ind_test = technical_indicators_normalised[n:]
    y_test = next_day_open_values_normalised[n:]
    unscaled_y_test= next_day_open_values[n:]

    lstm_input = Input(shape=(history_points,1), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
    #######
    model = lstm_model_tech(lstm_input,dense_input,ohlcv_train,tech_ind_train,y_train)
    #######
    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_predicted = model.predict([ohlcv_histories_normalised, technical_indicators_normalised])
    assert unscaled_y_test.shape == y_test_predicted.shape

    #print(y_test_predicted)
    assert unscaled_y_test.shape == y_test_predicted.shape

    mse = mean_squared_error(y_test,y_test_predicted)
    #print("R2",r2_score(y_test,y_test_predicted))
    mae = mean_absolute_error(y_test,y_test_predicted)


    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    #y_predicted = y_normaliser.inverse_transform(y_predicted)

    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100

    plt.plot(range(n,len(data)-51), unscaled_y_test[:-1], label='real')
    plt.plot(range(n,len(data)-51),y_test_predicted[:-1], label='predicted')
    plt.legend(['Real', 'Predicted'])

    plt.show()
    st.pyplot()
    return mse,mae,scaled_mse



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
    #df.to_csv(ticker + ".csv")
    #plt.show()
    #print()
    #print("##############################################################################")
    #print("Today's price")
    #today_stock = df.iloc[1:2]
    #print(today_stock)
    #print("##############################################################################")
    #print()
    return df

@st.cache(suppress_st_warning=True)
def summary_stats(ticker):
    df_summary = ticker
    return ticker

@st.cache(suppress_st_warning=True)
def ratio_indicators(ticker):
    df_ratios = fa.ratios(ticker)
    return df_ratios

def get_data_yahoo(ticker, start, end):
    data = pdr.get_data_yahoo(ticker, start= str(start), end = str(end) )
    return st.dataframe(data)



def plotData(ticker, start, end):

    df_stockdata = loadData(ticker, start, end)
    df_stockdata.index = pd.to_datetime(df_stockdata.index)


    set_pub()
    fig, ax = plt.subplots(2,1)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    ma1_checkbox = st.checkbox('Fast Moving Average')

    ma2_checkbox = st.checkbox('Slow Average 2')

    ax[0].set_title('Adj Close Price %s' % ticker, fontdict = {'fontsize' : 15})
    ax[0].plot(df_stockdata.index, df_stockdata.values,'g-',linewidth=1.6)
    ax[0].set_xlim(ax[0].get_xlim()[0] - 10, ax[0].get_xlim()[1] + 10)
    ax[0].grid(True)

    if ma1_checkbox:
        days1 = st.slider('Business Days to roll MA1', 5, 50, 1)
        ma1 = df_stockdata.rolling(days1).mean()
        ax[0].plot(ma1, 'b-', label = 'MA %s days'%days1)
        ax[0].legend(loc = 'best')
    if ma2_checkbox:
        days2 = st.slider('Business Days to roll MA2', 5, 200, 1)
        ma2 = df_stockdata.rolling(days2).mean()
        ax[0].plot(ma2, color = 'magenta', label = 'MA %s days'%days2)
        ax[0].legend(loc = 'best')

    ax[1].set_title('Daily Total Returns %s' % ticker, fontdict = {'fontsize' : 15})
    ax[1].plot(df_stockdata.index[1:], df_stockdata.pct_change().values[1:],'r-')
    ax[1].set_xlim(ax[1].get_xlim()[0] - 10, ax[1].get_xlim()[1] + 10)
    plt.tight_layout()
    ax[1].grid(True)
    st.pyplot()

def prophet():
    df =get_historical(from1,to1,timeframe)

    df = df.reset_index()
    dates = df['Date']

    m = Prophet()

    # df = df.drop(['1. open', '2. high', '3. low'], axis=1)
    df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
    m.fit(df)

    # Create Future dates
    future_prices = m.make_future_dataframe(periods=10)

    # Predict Prices
    forecast = m.predict(future_prices)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # Learn more Prophet tomorrow and plot the forecast for amazon.
    #fig = m.plot(forecast)
    #ax1 = fig.add_subplot(111)
    #ax1.set_title(ticker1, fontsize=16)
    #ax1.set_xlabel("Date", fontsize=12)
    #ax1.set_ylabel("Close Price", fontsize=12)

    #plt.show()
    #st.pyplot(fig)

    m = Prophet(changepoint_prior_scale=0.01).fit(df)
    future = m.make_future_dataframe(periods=300, freq=timeframe)
    fcst = m.predict(future)
    fig = m.plot(fcst)
    # plt.title("15 day prediction \n 1 month time frame")

    plt.show()
    st.pyplot(fig)

    fig = m.plot_components(fcst)
    plt.show()
    st.pyplot(fig)


def plotData1(ticker,start,end):
    df = loadData(ticker,start,end)

    df.index = pd.to_datetime(df.index)
    #st.write(df)
    #set_pub()
    #df['MA5'] = df.close.rolling(5).mean()
    #df['MA20'] = df.close.rolling(20).mean()

    #fig= df['Close'].plot()
    #plt.tight_layout()
    #plt.title(ticker1 + ': '+ timeframe)
    #df = df.to_csv('ticker1' + ".csv")
    #plt.grid(True)
    #plt.show()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))


    ma1_checkbox = st.checkbox('Fast Moving Average')

    ma2_checkbox = st.checkbox('Slow Moving Average')

    if ma1_checkbox:
        days1 = st.slider('Slow Moving Average', 5, 50, 10)
        ma1 = df['Close'].rolling(days1).mean()
        #fig(ma1, 'b-', label='MA %s points' % days1)
        #fig.legend(loc='best')
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(days1).mean(), line=dict(color='Black', width=1)))

    if ma2_checkbox:
        days2 = st.slider('Fast Moving Average', 5, 200, 30)
        ma2 = df['Close'].rolling(days2).mean()
        #fig.plot(ma2, color='magenta', label='MA %s points' % days2)
        #fig.legend(loc='best')
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(days2).mean(),line=dict(color='orange', width=1)))

    st.write('The plot of the stock ',ticker,'is displayed below ','from the start date of ',str(start),'to the end date of',str(end))
    st.plotly_chart(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

def rolling_sharpe(y):
    def geom_mean(y):
        y_1 = y + 1
        y_1_prod = np.prod(y_1)
        return y_1_prod**(1/(len(y))) - 1
    return np.sqrt(252) * (geom_mean(y) / y.std())


def plot_std_ret(ticker, start, end):
    def standard_ret(df):
        ret = df.pct_change()[1:]
        mean = ret.values.mean()
        std = ret.values.std()
        return (ret - mean) / std


    #from bokeh.plotting import figure

    #fig, ax = plt.subplots(figsize=(9,4))
    df_stockdata = loadData(ticker, start, end)['Adj Close']
    #ax.plot(df_stockdata.index[1:],standard_ret(df_stockdata).values)
    st.area_chart(standard_ret(df_stockdata).values)
    #ax.set_title('Standardized daily total returns %s'%ticker)
    #ax.set_xlim(ax.get_xlim()[0] - 10, ax.get_xlim()[1] + 10)
    #plt.grid(True)


def plot_trailing(ticker, start, end):
    ret = loadData(ticker, start, end).pct_change()[1:]
    days = st.slider('Business Days to roll', 5, 120, 30)
    trailing_median = ret.rolling(days).median()
    trailing_max = ret.rolling(days).max()
    trailing_min = ret.rolling(days).min()
    trailing_iqr = ret.rolling(days).apply(iqr)
    q3_rolling = ret.rolling(days).apply(lambda x: np.percentile(x,75))
    q1_rolling = ret.rolling(days).apply(lambda x: np.percentile(x,25))
    soglia_upper = trailing_iqr*1.5 + q3_rolling
    soglia_lower = q1_rolling - trailing_iqr*1.5
    trailing_all = pd.concat([trailing_median, trailing_max, trailing_min,trailing_iqr
                              ,soglia_upper, soglia_lower],
                             axis = 1)
    trailing_all.columns = ['Median', 'Max', 'Min','IQR','Q3 + 1.5IQR','Q1 - 1.5IQR']
    fig, ax = plt.subplots(figsize = (9,5))
    trailing_all.plot(ax = ax)
    ax.set_title('Rolling nonParametric Statistics (%s days)'%days
                     , pad = 30, fontdict = {'fontsize' : 17})
    ax.set_xlim(ax.get_xlim()[0] - 15, ax.get_xlim()[1] + 15)
    ax.legend(bbox_to_anchor=(0,0.96,0.96,0.2), loc="lower left",
                mode="expand", borderaxespad = 1, ncol = 6)
    ax.set_xlabel('')
    plt.grid(True)
    st.pyplot()

    ii = trailing_all.dropna().reset_index().drop('Date', axis = 1)
    st.subheader('Rolling nonParametric Statistics (%s days)'%days)
    print(st.dataframe(trailing_all.dropna()))

    st.subheader('Interactive chart, {} rolling observations,\
                     from {} to {}'.format(len(ii), parse(str(trailing_all.dropna().index[0])).date(),
                     parse(str(trailing_all.dropna().index[-1])).date()))
    st.line_chart(ii, width=800, height=120)

    ret = loadData(ticker, start, end).pct_change()[1:]
    trail_aim = trailing_all[['Q3 + 1.5IQR', 'Q1 - 1.5IQR']]

    def daterange(start_date, end_date):
        days_ = days
        for n in range(0, int ((end_date - start_date).days), days_):
            yield start_date + timedelta(n)

    def outliers():
        lista = []
        thresholds = []

        for i in range(len(ret)-days):
            ret_ = ret.iloc[i:days+i]
            trail_ = trail_aim.iloc[days+i]
            right_ret = np.where(ret_ > trail_['Q3 + 1.5IQR'], 1, 0).sum()
            left_ret = np.where(ret_ < trail_['Q1 - 1.5IQR'], 1, 0).sum()
            lista.append((right_ret, left_ret, (i,days+i)))
            trail_['Q1 - 1.5IQR'] = round(float(trail_['Q1 - 1.5IQR']),4)
            trail_['Q3 + 1.5IQR'] = round(float(trail_['Q3 + 1.5IQR']),4)
            thresholds.append((trail_['Q1 - 1.5IQR'], trail_['Q3 + 1.5IQR']))
        df = pd.DataFrame(np.random.randn(len(ret)-days,2))
        df.index = [elem[2] for elem in lista]

        df.columns = ['Left tail', 'Right tail']
        df['Right tail'] = [elem[0] for elem in lista]
        df['Left tail'] = [elem[1] for elem in lista]
        df['Thresholds'] = [t for t in thresholds]

        df2 = pd.DataFrame()
        df2['uno'] = [i[0] for i in df.index]
        df2['due'] = [i[1] for i in df.index]


        new = pd.DataFrame()
        new['A'] = [(str(ret.index[i].date()), str(ret.index[i+days].date())) for i in range(len(ret)-days)]
        new.index = new['A']

        df.index = new.index


        return df

    st.subheader('Number outliers for each datarange (%s business days)'%days)
    df_outliers = outliers()
    st.dataframe(df_outliers)
    st.subheader('Sorted by number of positive outliers (decreasing order)')
    st.dataframe(df_outliers.sort_values(by = 'Right tail', ascending = False))
    st.subheader('Sorted by number of negative outliers (decreasing order)')
    st.dataframe(df_outliers.sort_values(by = 'Left tail',ascending = False))


st.cache(suppress_st_warning=True)
def rolling_sharpe_plot(ticker, start, end):
    data_ = loadData(ticker, start, end)['Close']
    ret = data_.pct_change()[1:]
    start_sp = data_.index[0].strftime('%Y-%m-%d')
    sp500 = pdr.get_data_yahoo('^SP500TR', start= start_sp, end = str(end))
    sp500_ret = sp500['Close'].pct_change()[1:]

    days2 = st.slider('Business Days to roll', 5, 130, 50)
    rs = ret.rolling(days2).apply(rolling_sharpe)
    if len(rs.index) == len(sp500_ret):
        rs_sp500 = sp500_ret.rolling(days2).apply(rolling_sharpe)[:]
    else:
        rs_sp500 = sp500_ret.rolling(days2).apply(rolling_sharpe)[:-1]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(rs.index, rs.values, 'b-', label = 'Geometric Rolling Sharpe %s'%ticker)
    ax.plot(rs.index, rs_sp500, 'r-', label = 'Geometric Rolling Sharpe S&P500 (TR)')
    ax.set_title('Geometric Rolling Sharpe ratio (%s days, annualized)'%days2, fontdict = {'fontsize' : 15})
    ax.set_xlim(ax.get_xlim()[0] - 15, ax.get_xlim()[1] + 15)
    ax.legend(loc = 'best')
    plt.grid(True)
    st.pyplot()


sp500_list = pd.read_csv('SP500_list.csv')
Currency_list = pd.read_csv('Currency_list.csv')


#ticker = st.selectbox('Select the ticker if present in the S&P 500 index', sp500_list['Symbol'], index = 5).upper()
# ticker1 = st.selectbox('Select the following Forex ticker',Currency_list['Symbol'], index = 2)
# from1 = ticker1[0:3]
# to1 = ticker1[3:]
# pivot_sector = True

#checkbox_noSP = st.checkbox('Select this box to write the ticker (if not present in the S&P 500 list). \ Deselect to come back to the S&P 500 index stock list')
#if checkbox_noSP:
    #ticker = st.text_input('Write the ticker (check it in yahoo finance)', 'MN.MI').upper()


#start = st.text_input('Enter the start date in yyyy-mm-dd format:', '2018-01-01')
#end = st.text_input('Enter the end date in yyyy-mm-dd format:', '2019-01-01')

# timeframe = st.selectbox('Please enter the timeframe:',Currency_list['Timeframe'], index = 2)

@st.cache(suppress_st_warning=True)
def get_historical(from1,to1,timeframe):
    cc = ForeignExchange(key='N6A6QT6IBFJOPJ70', output_format='pandas')
    data, meta_data = cc.get_currency_exchange_intraday(from_symbol= from1.strip(), to_symbol=to1.strip(), interval= timeframe.strip(),outputsize='full')
    df = pd.DataFrame(data)
    #df= pd.date_range(pd.to_datetime('today'), periods=10, freq='15min')
    df= df.rename(columns={'date': 'Date', '1. open': 'Open','2. high':'High','3. low': 'Low','4. close':'Close'})
    df.index.names = ['Date']
    st.write(df)
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


def time_pred(timeframe):
    if (timeframe.strip() == '1min'):
        return 100
    elif (timeframe.strip() == '5min'):
        return 200
    elif (timeframe.strip() == '15min'):
        return 300
    elif (timeframe.strip() == '30min'):
        return 400
    elif (timeframe.strip() == '60min'):
        return 500
    elif (timeframe.strip() == '240min'):
        return 600

import numpy
def create_dataset(dataset, time_step=100):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

def reset_my_index(df):
  res = df[::-1].reset_index(drop=True)
  return(res)

class Tweet(object):

    def __init__(self, content, polarity):
        self.content = content
        self.polarity = polarity

def retrieving_tweets_polarity(symbol):
    consumer_key = 'E0pFYVai9VaOhqLiRBEC6gpGF'
    consumer_secret = 'XAMh4l9XL5nwFK3MN5tAjtXA2YgDN1tw5f7L2n6dz5ib8VYlbm'

    access_token = '3261604734-86c7DOJP98GwNeFWzvgPQKFUTyHn1ZFwlloJP3v'
    access_token_secret = 'eXEmlEAdxaFjueVP03jsAWeOeNMkI7ToiDQkyvLDa6eX7'
    nltk.download('punkt')
    num_of_tweets = int(300)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    user = tweepy.API(auth)

    tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='en', exclude_replies=True).items(
        num_of_tweets)

    tweet_list = []  # List of tweets alongside polarity
    global_polarity = 0  # Polarity of all tweets === Sum of polarities of individual tweets
    tw_list = []  # List of tweets only => to be displayed on web page
    # Count Positive, Negative to plot pie chart
    pos = 0  # Num of pos tweets
    neg = 1  # Num of negative tweets

    for tweet in tweets:
        count = 20  # Num of tweets to be displayed on web page
        # Convert to Textblob format for assigning polarity
        tw2 = tweet.full_text
        tw = tweet.full_text
        # Clean
        tw = api.clean(tw)
        # print("-------------------------------CLEANED TWEET-----------------------------")
        # print(tw)
        # Replace &amp; by &
        tw = re.sub('&amp;', '&', tw)
        # Remove :
        tw = re.sub(':', '', tw)
        # print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
        # print(tw)
        # Remove Emojis and Hindi Characters
        tw = tw.encode('ascii', 'ignore').decode('ascii')

        # print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
        # print(tw)
        blob = TextBlob(tw)
        polarity = 0  # Polarity of single individual tweet
        for sentence in blob.sentences:

            polarity += sentence.sentiment.polarity
            if polarity > 0:
                pos = pos + 1
            if polarity < 0:
                neg = neg + 1

            global_polarity += sentence.sentiment.polarity
        if count > 0:
            tw_list.append(tw2)

        tweet_list.append(Tweet(tw, polarity))
        count = count - 1
    global_polarity = global_polarity / len(tweet_list)
    neutral = num_of_tweets - pos - neg
    if neutral < 0:
        neg = neg + neutral
        neutral = 20
    print()
    print("##############################################################################")
    print("Positive Tweets :", pos, "Negative Tweets :", neg, "Neutral Tweets :", neutral)
    print("##############################################################################")
    print()
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    explode = (0, 0, 0)
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    # plt.savefig('static/SA.png')
    #plt.close(fig)
    plt.show()

    st.pyplot()

    if global_polarity > 0:
        #print()
        #print("##############################################################################")
        print("Tweets Polarity: Overall Positive")
        #print("##############################################################################")
        #print()
        tw_pol = "Overall Positive"
    else:
        #print()
        #print("##############################################################################")
        print("Tweets Polarity: Overall Negative")
        #print("##############################################################################")
        #print()
        tw_pol = "Overall Negative"



def recommending(df, global_polarity, today_stock, mean=1.5):
    df = get_historical(from1,to1,timeframe)
    if df.iloc[-1]['Close'] < mean:
        if global_polarity > 0:
            #print()

            idea = "RISE"
            decision = "BUY"
            #print()
            #print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea, "in", "USDCAD",
                  "stock is expected => ", decision)
        elif global_polarity < 0:
            #print()
            idea = "FALL"
            decision = "SELL"
            #print()
            #print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea, "in", "USDCAD",
                 "stock is expected => ", decision)
    else:
        #print()
        idea = "FALL"
        decision = "SELL"
        #print()
        #print("##############################################################################")
        print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea, "in", "USDCAD",
              "stock is expected => ", decision)

    return idea, decision



@st.cache(suppress_st_warning=True)
def def_model_lstm_GRU(X_train,y_train,X_test, ytest):
    model = Sequential()
    model.add(LSTM(10, return_sequences=True,input_shape=(50, 1)))
    model.add(GRU(10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',)
    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=30, batch_size=64, verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    return train_predict,test_predict,model


def lstm():
    from sklearn.preprocessing import MinMaxScaler
    df = loadData(ticker,start,end)
    df = df['Close']

    #Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

    training_size = int(len(df1) * 0.7)

    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]


    time_step = 50 #time_pred(timeframe)
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)


    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    import time

    @st.cache(suppress_st_warning=True)
    def def_model_lstm_stacked(X_train,y_train,X_test, ytest):
        model = Sequential()
        model.add(LSTM(10, return_sequences=True,input_shape=(50, 1)))
        model.add(LSTM(10))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=30,batch_size=30,verbose=1)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        return train_predict,test_predict,model

    train_predict,test_predict,model= def_model_lstm_stacked(X_train,y_train,X_test, ytest)

    # model = Sequential()
    # model.add(LSTM(10, return_sequences=True,input_shape=(50, 1)))
    # model.add(LSTM(10))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam',)
    # model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=1, batch_size=64, verbose=1)
    # train_predict = model.predict(X_train)
    # test_predict = model.predict(X_test)

    import math

    mse= mean_squared_error(y_train, train_predict)
    mae = mean_absolute_error(ytest, test_predict)
    real_mse = np.mean(np.square(ytest - test_predict))
    scaled_mse = real_mse / (np.max(ytest) - np.min(ytest)) * 100

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)



    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(1,1,1)
    look_back = 50 #time_pred(timeframe)
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
    # plot baseline and predictions


    ax.plot(scaler.inverse_transform(df1))
    ax.plot(trainPredictPlot)
    ax.plot(loc='best')
    ax.plot(testPredictPlot)
    ax.grid(True)
    st.pyplot(fig)
    #plt.show()
    #plt.tight_layout()
    #st.pyplot()


    # x_input = test_data[len(test_data) - 50:].reshape(1, -1)
    # temp_input = list(x_input)
    # temp_input = temp_input[0].tolist()
    #
    #
    # lst_output = []
    # n_steps = 50
    # i = 0
    # while (i < 50):
    #
    #     if (len(temp_input) > n_steps):
    #         # print(temp_input)
    #         x_input = np.array(temp_input[1:])
    #         # print("{} 15 mins input {}".format(i,x_input))
    #         x_input = x_input.reshape(1, -1)
    #         x_input = x_input.reshape((1, n_steps, 1))
    #         # print(x_input)
    #         yhat = model.predict(x_input, verbose=0)
    #         # print("{} 15 mins output {}".format(i,yhat))
    #         temp_input.extend(yhat[0].tolist())
    #         temp_input = temp_input[1:]
    #         # print(temp_input)
    #         lst_output.extend(yhat.tolist())
    #         i = i + 1
    #     else:
    #         x_input = x_input.reshape((1, n_steps, 1))
    #         yhat = model.predict(x_input, verbose=0)
    #         # print(yhat[0])
    #         temp_input.extend(yhat[0].tolist())
    #         # print(len(temp_input))
    #         lst_output.extend(yhat.tolist())
    #         i = i + 1
    #
    # # print(lst_output)
    #
    # #day_new = np.arange(1, 101)
    # #day_pred = np.arange(101, 131)
    #
    # df3 = df1.tolist()
    # df3.extend(lst_output)
    #
    # predictions = scaler.inverse_transform(lst_output)
    # mid = len(predictions) / 2
    #
    # if ((predictions[0] > predictions[int(mid)]) and (predictions[0] > predictions[-1])):
    #     out1= str("SELL @ %3f" % predictions[0])
    #     exit1 = str("TAKE PROFIT @ %3f" % predictions[49])
    # elif ((predictions[0] < predictions[int(mid)] and predictions[0] < predictions[-1])):
    #     out1 = str("BUY @ %3f" % predictions[0])
    #     exit1 = str("TAKE PROFIT @ %3f" % predictions[49])
    # else:
    #     out1= "HOLD"
    #     exit1 = 'HOLD'
    #
    # x1 = len(df3) - 50
    # history = scaler.inverse_transform(df3[:-50])
    # prediction = scaler.inverse_transform(df3[int(x1):])

    # day_new = np.arange(0, len(df1))
    # day_pred = np.arange(len(df1), len(df1) + 50)
    #
    #
    # plt.plot(day_new, history, label='Previous')
    # plt.plot(day_pred, prediction, label='Prediction')
    # plt.show()
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.tight_layout()
    # st.pyplot()

    return mse,mae,scaled_mse


def lstm_GRU():
    from sklearn.preprocessing import MinMaxScaler
    df = loadData(ticker,start,end)
    df = df['Close']

    #Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

    training_size = int(len(df1) * 0.7)

    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]


    time_step = 50 #time_pred(timeframe)
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)


    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    train_predict,test_predict,model= def_model_lstm_GRU(X_train,y_train,X_test, ytest)

    # model = Sequential()
    # model.add(LSTM(10, return_sequences=True,input_shape=(50, 1)))
    # model.add(LSTM(10))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam',)
    # model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=1, batch_size=64, verbose=1)
    # train_predict = model.predict(X_train)
    # test_predict = model.predict(X_test)

    import math
    from sklearn.metrics import mean_squared_error,mean_absolute_error
    import matplotlib.pyplot as plt
    mse= mean_squared_error(y_train, train_predict)
    mae = mean_absolute_error(ytest, test_predict)
    real_mse = np.mean(np.square(ytest - test_predict))
    scaled_mse = real_mse / (np.max(ytest) - np.min(ytest)) * 100

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)



    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(1,1,1)
    look_back = 50 #time_pred(timeframe)
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
    # plot baseline and predictions


    ax.plot(scaler.inverse_transform(df1))
    ax.plot(trainPredictPlot)
    ax.plot(loc='best')
    ax.plot(testPredictPlot)
    ax.grid(True)
    st.pyplot(fig)

    return mse,mae,scaled_mse

''' # Stock Market Prediction Dashboard '''

sp500_list = pd.read_csv('SP500_list.csv')

ticker = st.selectbox('Select ticker (present in the S&P 500 index)', sp500_list['Symbol'], index = 50).upper()
pivot_sector = True

checkbox_noSP = st.checkbox('If not present in the S&P 500 list, select this box. \
                            Deselect in order to come back to the S&P 500 index stock list')

if checkbox_noSP:
    ticker1 = st.text_input('Enter the ticker (not present in S&P 500 index)', 'MN.MI').upper()
    ticker = ticker1
    pivot_sector = False


start = st.text_input('Enter the start date in yyyy-mm-dd format:', '2000-01-01')
end = st.text_input('Enter the end date in yyyy-mm-dd format:', datetime.today().strftime('%Y-%m-%d'))



try:
    start = parse(start).date()
    #print('The start date is valid')
    control_date1 = True
except ValueError:
    st.error('Invalid Start date')
    control_date1 = False

try:
    end = parse(end).date()
    #print('The end date is valid')
    control_date2 = True
except ValueError:
    st.error('Invalid End date')
    control_date2 = False

def check_dates():
    return control_date1 & control_date2


if start <= datetime(1970,1,1,0,0).date():
    st.error('Please insert a date posterior to 1st January 1970')
    pivot_date = False
else:
    pivot_date = True

if check_dates() and pivot_date == True:

    if len(loadData(ticker, start, end)) > 0: # if the ticker is invalid the function returns an empty series

        image = Image.open('Image.jpeg')

        st.sidebar.image(image, caption='',use_column_width=True)
        st.sidebar.header('Stock Prediction Dashboard')
        st.sidebar.subheader('Choose option to visualize')

        ticker_meta = yf.Ticker(ticker)

        series_info  = pd.Series(ticker_meta.info,index = reversed(list(ticker_meta.info.keys())))
        series_info = series_info.loc[['symbol', 'shortName','exchange', 'exchangeTimezoneName', 'marketCap', 'quoteType']]
        if pivot_sector:
            sector = sp500_list[sp500_list['Symbol'] == ticker]['Sector']
            sector = sector.values[0]
            series_info['sector'] = sector

        series_info.name = 'Stock'


        principal_graphs_checkbox = st.sidebar.checkbox('Stock Visualisation', value = True)
        if principal_graphs_checkbox:
            st.title('Stock Visualisation')
            st.dataframe(series_info)
            plotData1(ticker, start, end)

        trailing_checkbox = st.sidebar.checkbox('Historical Prices and Volume')
        if trailing_checkbox:
            st.title('Historical Prices and Volume ')
            st.markdown('''The acquired dataset is shown below as a table: ''')
            get_data_yahoo(ticker, start, end)

        std_ret_checkbox = st.sidebar.checkbox('Standardized daily total returns')
        if std_ret_checkbox:
            st.title('Standardized daily total returns')
            st.markdown('''The daily total return in % is shown as a graph''')
            plot_std_ret(ticker, start, end)

        rs_checkbox = st.sidebar.checkbox('Rolling Sharpe ratio vs Rolling Sharpe ratio S&P500, (annualized)')

        if rs_checkbox:
            st.title('Rolling Sharpe Ratio')
            st.markdown('''It is a useful technique to compare the historical performance of a fund. 
            The is because RSR gives investors a time-varying performance
            The geometric rolling sharpe ratio(RSR) of the stock is compared with the geometric rolling sharpe ratio of S&P500 index.
            The RSR is calculated by fixing the risk free rate equal to 0. The formula is shown below:
            ''')
            image1 = Image.open('Sharpe.png')
            st.image(image1)
            rolling_sharpe_plot(ticker, start, end)


        fundamental_checkbox = st.sidebar.checkbox('Stacked LSTM Model')
        if fundamental_checkbox:
            st.title('Stacked LSTM Model')
            image2 = Image.open('LSTM_Stacked.jpeg')
            st.image(image2,caption='Model Architecture')
            st.markdown('''Closing price feature is passed as input to the first LSTM layer, LSTM_0. LSTM_0 neurons collect data, and a weighted value is generated that is passed to the second LSTM layer, LSTM_1. After a weighted value is generated along the path from LSTM_1, it is passed to the dense layer,dense_out. A weighted value is also generated in the dense layer and is used to produce the output. Weighted value is passed to the output neuron and weight is generated accordingly. 
The output is compared with the original value for the calculation of the error function. Weighted values are updated until the model reaches a minimum cost.
''')
            mse,mae,adjuted_mse = lstm()

            '''## Mean Squared Error(MSE): ''' + str(mse)
            '''## Mean Absolute Error(MAE): ''' + str(mae)
            '''## Adjusted MSE: ''' + str(adjuted_mse)

        fundamental_checkbox1 = st.sidebar.checkbox('LSTM-GRU Hybrid Model')
        if fundamental_checkbox1:
            st.title('LSTM-GRU Hybrid Model')
            image3 = Image.open('LSTM_GRU.jpeg')
            st.image(image3,caption='Model Architecture')
            st.markdown('''Closing price feature is passed as input to the first LSTM layer. The LSTM_0 layer neurons collect data and a weighted value is generated that is passed to the GRU_0 layer. After a weighted value is generated along the path from the GRU_0 layer, it is passed to the dense layer. A weighted value is also generated in the dense layer and is used to produce the output. Weighted value is passed to the output neuron and weight is generated accordingly.  ''')

            mse,mae,adjuted_mse = lstm_GRU()
            '''## Mean Squared Error(MSE): ''' + str(mse)
            '''## Mean Absolute Error(MAE): ''' + str(mae)
            '''## Adjusted MSE: ''' + str(adjuted_mse)

        fundamental_checkbox2 = st.sidebar.checkbox('LSTM-Technical Indicator Hybrid Model')
        if fundamental_checkbox2:
            st.title('LSTM-Technical Indicator Hybrid Model')
            image4 = Image.open('LSTM_Tech.jpeg')
            st.image(image4,caption='Model Architecture')
            st.markdown('''Closing price feature is passed as input to the first LSTM layer, LSTM_0, while the technical indicator is passed to the tech_input layer in the other branch. LSTM_1 neuron collects data, and a weighted value is generated that is passed to the second LSTM layer, LSTM_2. A dropout layer is added,LSTM_dropout that is added to prevent overfitting and is passed to the concatenation layer.
On the other branch, the MACD indicator values are added to the tech_input layer. After a weighted value is generated along the path from the tech_input layer, it is passed to the dense layer, tech_dense_0. A weighted value is also generated in the tech_dense_0 layer and is used to produce the output. It is then passed to an activation function of sigmoid as well as a dropout layer.
The outputs are combined in the dense_pooling layer and then the dense_out layer. This output is compared with the original value for the calculation of error function. Weighted values are updated until the model reaches a minimum cost.
''')
            mse,mae,adjuted_mse = tech_ind()

            '''## Mean Squared Error(MSE): ''' + str(mse)
            '''## Mean Absolute Error(MAE): ''' + str(mae)
            '''## Adjusted MSE: ''' + str(adjuted_mse)

    else:
        st.error('Invalid ticker')







