# Stock Market Prediction using Hybrid LSTM Models

In the ever-evolving world of financial markets, stock market movement predictions are one of the most passionately discussed topics in the modern world. More than half of the US households have investments in the stock market, showing the role it plays in our daily lives. However, due to its volatile nature, it is very difficult to predict future price outcomes. Traders constantly are on the lookout for models and techniques to outperform the market and earn higher profits. Thus, researchers are constantly exploring unique forecasting techniques.


This paper explores 3 hybrid LSTM based models: Stacked LSTM(Baseline), LSTM-GRU and LSTM-Technical Indicator model and has been trained on ‘Berkshire Hathaway (BRK-B)’ stock with 60-minute intervals obtained from Yahoo Finance API. The data was passed through the models and the performance of the models was evaluated on 3 metrics of loss: mean squared errors (MSE), mean absolute error (MAE) and Adjusted MSE. The experiment also includes the fine tuning of hyperparameters to determine the best performing models.
## Installation

pip install -r requirements.txt 

```bash
1. pip install -r requirements.txt
2. streamlit run app.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
