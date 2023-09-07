import matplotlib.pyplot as plt
import pandas as pd
# Assuming 'data' is your DataFrame and it has been processed accordingly
data = pd.read_csv('pltr_with_features.csv')

plt.figure(figsize=(14,7))

# Plot close price
plt.plot(data['Close'], label='Close Price', color='blue')

import numpy as np

# Find indices where trend changes to uptrend
uptrend_indices = data[(data['Target'].shift() == 0) & (data['Target'] == 1) & 
                       (data['Target'].shift(-1) == 1) & (data['Target'].shift(-3) == 1)].index

# Adjust indices to point to the last occurrence of the 3 continuous 1's
uptrend_indices = np.array([index+2 for index in uptrend_indices if index+2 < len(data)])

plt.scatter(uptrend_indices, data.loc[uptrend_indices, 'Close'], color='green', label='Uptrend Signal')

# Find indices where trend changes to downtrend
downtrend_indices = data[(data['Target'].shift() == 0) & (data['Target'] == -1) & 
                         (data['Target'].shift(-1) == -1) & (data['Target'].shift(-3) == -1)].index

# Adjust indices to point to the last occurrence of the 3 continuous -1's
downtrend_indices = np.array([index+2 for index in downtrend_indices if index+2 < len(data)])

plt.scatter(downtrend_indices, data.loc[downtrend_indices, 'Close'], color='red', label='Downtrend Signal')

plt.title('BTC-USD Close Price with Trend Signals')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
