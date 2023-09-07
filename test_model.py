import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
# Load the saved model
best_model = joblib.load('best_gb_model.pkl')

# Load data
new_data = pd.read_csv('eth_with_features.csv')

# Preprocess new data (assuming it's stored in a DataFrame named `new_data`)
new_X = new_data[['Close', 'Candlestick Shape', 'Candlestick Pattern', '200 MA', 'Primary Trend', 'RSI', 'Direction']]
new_X = pd.get_dummies(new_X, columns=['Candlestick Shape', 'Candlestick Pattern', 'Primary Trend'])
# Scale numerical features
scaler = StandardScaler()
new_X[['Close', '200 MA', 'RSI']] = scaler.fit_transform(new_X[['Close', '200 MA', 'RSI']])

# Make predictions on the new data
new_y_pred = best_model.predict(new_X)

# Get the target variable from the new_data DataFrame
new_y_true = new_data['Target']
# Calculate the classification report
report = classification_report(new_y_true, new_y_pred)

# Print the classification report
print("Classification Report:")
print(report)
# Get the features for the last day in your dataset
last_day_features = new_X.iloc[-1].values.reshape(1, -1)

# Predict the target for tomorrow
predicted_target = best_model.predict(last_day_features)[0]
# Print the predicted target
print('The predicted target for tomorrow is:', predicted_target)
# Add predictions and true target values to the testing set
new_X['Prediction'] = new_y_pred
new_X['True_Target'] = new_y_true

# Export testing set with predictions and true target values to a CSV file
new_X.to_csv('ETH_Test_Predict.csv', index=False)
 