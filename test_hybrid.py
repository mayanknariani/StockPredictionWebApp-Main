import pandas as pd
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

ticker_meta = yf.Ticker('AAPL')

series_info  = pd.Series(ticker_meta.info,index = reversed(list(ticker_meta.info.keys())))
series_info = series_info.loc[['symbol', 'shortName','exchange',
                  'exchangeTimezoneName', 'marketCap', 'quoteType']]
