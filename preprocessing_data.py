import pandas as pd
import talib

# Load data
data = pd.read_csv('eth_usd.csv')

# Add RSI(14) feature
rsi_14 = talib.RSI(data['Close'], timeperiod=14)
data['RSI_14'] = rsi_14

# Add WT(10) and WT(11) features
wt_10 = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=10)
data['WT_10'] = wt_10

wt_11 = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=11)
data['WT_11'] = wt_11

# Add CCI(20) and CCI(1) features
cci_20 = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=20)
data['CCI_20'] = cci_20

cci_2 = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=2)
data['CCI_2'] = cci_2

# Add ADX(20) and ADX(2) features
adx_20 = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=20)
data['ADX_20'] = adx_20

adx_2 = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=2)
data['ADX_2'] = adx_2

# Add RSI(9) and RSI(1) features
rsi_9 = talib.RSI(data['Close'], timeperiod=9)
data['RSI_9'] = rsi_9

rsi_2 = talib.RSI(data['Close'], timeperiod=2)
data['RSI_2'] = rsi_2




# Calculate differences
close_open_diff = data['Close'] - data['Open']
high_low_diff = data['High'] - data['Low']

# Add Candlestick Shape column
data.loc[(close_open_diff > 0) & (high_low_diff > 0), 'Candlestick Shape'] = 'Bullish'
data.loc[(close_open_diff < 0) & (high_low_diff > 0), 'Candlestick Shape'] = 'Bearish'
data.loc[high_low_diff == 0, 'Candlestick Shape'] = 'Neutral'

# Calculate candlestick patterns
doji = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
engulfing = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
hammer = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
hanging_man = talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])

# Add Candlestick Pattern column
data.loc[doji > 0, 'Candlestick Pattern'] = 'Doji'
data.loc[engulfing > 0, 'Candlestick Pattern'] = 'Engulfing'
data.loc[hammer > 0, 'Candlestick Pattern'] = 'Hammer'
data.loc[hanging_man > 0, 'Candlestick Pattern'] = 'Hanging Man'
data.loc[(doji == 0) & (engulfing == 0) & (hammer == 0) & (hanging_man == 0), 'Candlestick Pattern'] = 'None'

# Refine labels using Candlestick Shape and Candlestick Pattern
data.loc[(data['Candlestick Shape'] == 'Bullish') & (data['Candlestick Pattern'] == 'Doji'), 'Direction'] = 0
data.loc[(data['Candlestick Shape'] == 'Bearish') & (data['Candlestick Pattern'] == 'Doji'), 'Direction'] = 0
data.loc[(data['Candlestick Shape'] == 'Bullish') & (data['Candlestick Pattern'] == 'Engulfing'), 'Direction'] = 1
data.loc[(data['Candlestick Shape'] == 'Bearish') & (data['Candlestick Pattern'] == 'Engulfing'), 'Direction'] = -1
data.loc[(data['Candlestick Shape'] == 'Bullish') & (data['Candlestick Pattern'] == 'Hammer'), 'Direction'] = 1
data.loc[(data['Candlestick Shape'] == 'Bearish') & (data['Candlestick Pattern'] == 'Hanging Man'), 'Direction'] = 1

# Add Price Change column
data['Price Change'] = data['Close'].diff()

# Identify primary trend
data['200 MA'] = talib.SMA(data['Close'], timeperiod=200)
data.loc[data['Close'] > data['200 MA'], 'Primary Trend'] = 'Bullish'
data.loc[data['Close'] < data['200 MA'], 'Primary Trend'] = 'Bearish'
# Confirm/reject trend with momentum indicators
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data.loc[(data['Primary Trend'] == 'Bullish') & (data['RSI'] > 50), 'Direction'] = 1
data.loc[(data['Primary Trend'] == 'Bearish') & (data['RSI'] < 50), 'Direction'] = -1
# Confirm/reject trend with other indicators

data.loc[(data['Direction'] == 'up') & (doji > 0), 'Direction'] = 0
# Remove rows with NaN values
data = data.dropna()
# Move "Direction" column to the end
direction_col = data.pop('Direction')
data.insert(len(data.columns), 'Direction', direction_col)
data = data.reset_index(drop=True)

# Add Trend Change column
data['Target'] = 0

for i in range(len(data) - 3):
    if data.loc[i, 'Direction'] == -1 and data.loc[i+1:i+3, 'Direction'].sum() == -3:
        data.loc[i+3, 'Target'] = -1  # Change to downtrend
    elif data.loc[i, 'Direction'] == 1 and data.loc[i+1:i+3, 'Direction'].sum() == 3:
        data.loc[i+3, 'Target'] = 1  # Change to uptrend




# Save data
data.to_csv('eth_with_features.csv', index=False)

print(data.head(10))



