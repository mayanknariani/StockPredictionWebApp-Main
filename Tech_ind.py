from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import pandas as pd
from sklearn import preprocessing
import numpy as np
from app import *

@st.cache(suppress_st_warning=True)
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

    from keras.models import Model
    from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
    from keras import optimizers
    import numpy as np
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    np.random.seed(4)
    import tensorflow
    tensorflow.random.set_seed(4)

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

    model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=10, epochs=10, shuffle=True, validation_split=0.3)
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

    real = plt.plot(range(n,len(data)-51), unscaled_y_test[:-1], label='real')
    pred = plt.plot(range(n,len(data)-51),y_test_predicted[:-1], label='predicted')
    plt.legend(['Real', 'Predicted'])

    plt.show()
    st.pyplot()
    return mse,mae,scaled_mse


