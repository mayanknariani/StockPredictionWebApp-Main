import FundamentalAnalysis as fa
import streamlit as st
import matplotlib.pyplot as plt, pandas as pd, numpy as np
import matplotlib
from PIL import Image
matplotlib.use('Agg')
#from fbprophet import Prophet
#import plotly.graph_objects as go
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
import re


def set_pub():
    rc('font', weight='bold')    # bold fonts are easier to see
    rc('grid', c='0.5', ls='-', lw=0.5)
    rc('figure', figsize = (10,8))
    plt.style.use('bmh')
    rc('lines', linewidth=1.3, color='b')

@st.cache(suppress_st_warning=True)
def loadData(ticker, start, end):
     df_stockdata = pdr.get_data_yahoo(ticker, start= str(start), end = str(end) )['Adj Close']
     df_stockdata.index = pd.to_datetime(df_stockdata.index)
     return df_stockdata

@st.cache(suppress_st_warning=True)
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

    ma1_checkbox = st.checkbox('Moving Average 1')

    ma2_checkbox = st.checkbox('Moving Average 2')

    ax[0].set_title('Adj Close Price %s' % ticker, fontdict = {'fontsize' : 15})
    ax[0].plot(df_stockdata.index, df_stockdata.values,'g-',linewidth=1.6)
    ax[0].set_xlim(ax[0].get_xlim()[0] - 10, ax[0].get_xlim()[1] + 10)
    ax[0].grid(True)

    if ma1_checkbox:
        days1 = st.slider('Business Days to roll MA1', 5, 150, 30)
        ma1 = df_stockdata.rolling(days1).mean()
        ax[0].plot(ma1, 'b-', label = 'MA %s days'%days1)
        ax[0].legend(loc = 'best')
    if ma2_checkbox:
        days2 = st.slider('Business Days to roll MA2', 5, 150, 30)
        ma2 = df_stockdata.rolling(days2).mean()
        ax[0].plot(ma2, color = 'magenta', label = 'MA %s days'%days2)
        ax[0].legend(loc = 'best')

    ax[1].set_title('Daily Total Returns %s' % ticker, fontdict = {'fontsize' : 15})
    ax[1].plot(df_stockdata.index[1:], df_stockdata.pct_change().values[1:],'r-')
    ax[1].set_xlim(ax[1].get_xlim()[0] - 10, ax[1].get_xlim()[1] + 10)
    plt.tight_layout()
    ax[1].grid(True)
    st.pyplot()

@st.cache(suppress_st_warning=True)
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


def plotData1():
    df = get_historical(from1,to1,timeframe)

    df.index = pd.to_datetime(df.index)

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

    ma1_checkbox = st.checkbox('Moving Average 1')

    ma2_checkbox = st.checkbox('Moving Average 2')

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

    fig, ax = plt.subplots(figsize=(9,4))
    df_stockdata = loadData(ticker, start, end)
    ax.plot(df_stockdata.index[1:],
          standard_ret(df_stockdata).values)
    ax.set_title('Standardized daily total returns %s'%ticker, fontdict = {'fontsize' : 15})
    ax.set_xlim(ax.get_xlim()[0] - 10, ax.get_xlim()[1] + 10)
    plt.grid(True)
    st.pyplot()


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



def rolling_sharpe_plot(ticker, start, end):
    data_ = loadData(ticker, start, end)
    ret = data_.pct_change()[1:]
    start_sp = data_.index[0].strftime('%Y-%m-%d')
    sp500 = pdr.get_data_yahoo('^SP500TR', start= start_sp, end = str(end) )
    sp500_ret = sp500['Close'].pct_change()[1:]

    days2 = st.slider('Business Days to roll', 5, 130, 50)
    rs_sp500 = sp500_ret.rolling(days2).apply(rolling_sharpe)
    rs = ret.rolling(days2).apply(rolling_sharpe)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(rs.index, rs.values, 'b-', label = 'Geometric Rolling Sharpe %s'%ticker)
    ax.plot(rs.index, rs_sp500, 'r-', label = 'Geometric Rolling Sharpe S&P500 (TR)')
    ax.set_title('Geometric Rolling Sharpe ratio (%s days, annualized)'%days2, fontdict = {'fontsize' : 15})
    ax.set_xlim(ax.get_xlim()[0] - 15, ax.get_xlim()[1] + 15)
    ax.legend(loc = 'best')
    plt.grid(True)
    st.pyplot()


''' # Summary of results'''

sp500_list = pd.read_csv('SP500_list.csv')
Currency_list = pd.read_csv('Currency_list.csv')


#ticker = st.selectbox('Select the ticker if present in the S&P 500 index', sp500_list['Symbol'], index = 5).upper()
ticker1 = st.selectbox('Select the following Forex ticker',Currency_list['Symbol'], index = 2)
from1 = ticker1[0:3]
to1 = ticker1[3:]
pivot_sector = True

#checkbox_noSP = st.checkbox('Select this box to write the ticker (if not present in the S&P 500 list). \ Deselect to come back to the S&P 500 index stock list')
#if checkbox_noSP:
    #ticker = st.text_input('Write the ticker (check it in yahoo finance)', 'MN.MI').upper()


#start = st.text_input('Enter the start date in yyyy-mm-dd format:', '2018-01-01')
#end = st.text_input('Enter the end date in yyyy-mm-dd format:', '2019-01-01')

timeframe = st.selectbox('Please enter the timeframe:',Currency_list['Timeframe'], index = 2)

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
def create_dataset(dataset, time_step=300):
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
        tw = p.clean(tw)
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
    return global_polarity,pos, neg, neutral


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
def lstm():
    from sklearn.preprocessing import MinMaxScaler
    df = get_historical(from1,to1,timeframe)
    df.index = pd.to_datetime(df.index)
    #df = reset_my_index(df)

    #Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

    training_size = int(len(df1) * 0.7)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    time_step = time_pred(timeframe)
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    #print(X_train.shape), print(y_train.shape)
    #print(X_test.shape), print(ytest.shape)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(40, return_sequences=False,input_shape=(300, 1)))
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
    look_back = time_pred(timeframe)
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
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
    st.pyplot()


    x_input = test_data[len(test_data) - time_pred(timeframe):].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()


    lst_output = []
    n_steps = time_pred(timeframe)
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

    day_new = np.arange(0, len(df1))
    day_pred = np.arange(len(df1), len(df1) + 300)


    plt.plot(day_new, history, label='Previous')
    plt.plot(day_pred, prediction, label='Prediction')
    plt.show()
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot()

    st.text(out1)
    st.text(exit1)

#try:
    #start = parse(start).date()
    #print('The start date is valid')
    #ontrol_date1 = True
#except ValueError:
    #st.error('Invalid Start date')
    #control_date1 = False


#try:
    #end = parse(end).date()
    #print('The end date is valid')
    #control_date2 = True
#except ValueError:
    #st.error('Invalid End date')
    #control_date2 = False

#def check_dates():
 #   return control_date1 & control_date2


#if start <= datetime(2000,1,1,0).date():
    #st.error('Please insert a date posterior to 1st January 1970')
    #pivot_date = False
    #else:
    #pivot_date = True


#if check_dates() and pivot_date == True:


    #if len(loadData(ticker, start, end)) > 0: # if the ticker is invalid the function returns an empty series


image = Image.open('imageforapp2.jpg')

st.sidebar.image(image, caption='', use_column_width=True)

st.sidebar.header('Stock prediction analysis')

st.sidebar.subheader('Please select option to visualise ')

#ticker_meta = yf.Ticker(ticker)

#series_info  = pd.Series(ticker_meta.info)
#series_info= series_info.reindex(reversed(list(ticker_meta.info.keys())))
#series_info = series_info.loc[['symbol', 'shortName', 'financialCurrency','exchange',
                  #'fullExchangeName', 'exchangeTimezoneName', 'marketCap', 'quoteType']]
#if pivot_sector:
    #sector = sp500_list[sp500_list['Symbol'] == ticker]['Sector']
    #sector = sector.values[0]
    #series_info['sector'] = sector


#series_info.name = 'Stock'
#st.dataframe(series_info)

principal_graphs_checkbox = st.sidebar.checkbox('Stock selection: ', value = True)

principal_graphs_checkbox1 = st.button('Submit')
if principal_graphs_checkbox1 or principal_graphs_checkbox:
    plotData1()

std_ret_checkbox = st.sidebar.checkbox('LSTM Prediction',value = True)

if std_ret_checkbox:
    st.title('LSTM Prediction')
    #st.warning('Calculating .... ')
    st.subheader(lstm())

std_ret_checkbox1 = st.sidebar.checkbox('Facebook Prophet Prediction',value = True)
if std_ret_checkbox1:
    st.title('Facebook Prophet Prediction')
    st.subheader(prophet())

trailing_checkbox = st.sidebar.checkbox('Sentiment Analysis',value=True)
if trailing_checkbox:
    st.title('Twitter Sentiment Analysis')
    st.subheader(retrieving_tweets_polarity(ticker1))


    #st.subheader('Interquartile range : Q3 - Q1')
    #st.subheader('Upper threshold : Q3 + 1.5IQR')
    #st.subheader('Lower threshold : Q1 - 1.5IQR')
    #st.write('')
    #plot_trailing(ticker, start, end)

# fundamental_checkbox = st.sidebar.checkbox('Fundamental Analysis')
# if fundamental_checkbox:
    #''' ## Fundamental analysis '''
   # st.title('Summary')
    #st.dataframe(summary_stats(ticker))

  #  st.title('Ratios and indicators')
  #  st.dataframe(ratio_indicators(ticker))

#rs_checkbox = st.sidebar.checkbox('Rolling Sharpe ratio vs Rolling Sharpe ratio S&P500, (annualized)')
# if rs_checkbox:
 #   ''' # Rolling Sharpe Ratio '''
#     ''' We compare the geometric rolling sharpe ratio (RSR) of the stock with the geometric rolling sharpe ratio of S&P500 (TR).
#    We calculate the RSR by fixing the risk free rate equal to 0.
#    Hence *RSR = rolling_returns_mean / rolling_returns_std*.
##    '''
 #   rolling_sharpe_plot(ticker, start, end)

#historical_prices_checkbox = st.sidebar.checkbox('Historical prices and volumes')
#if historical_prices_checkbox:
 #   st.title('Historical prices and volumes')
  #  get_data_yahoo(ticker, start, end)

    #else:
        #st.error('Invalid ticker')







