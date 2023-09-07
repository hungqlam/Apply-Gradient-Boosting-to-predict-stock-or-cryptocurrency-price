import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('btc_with_features.csv')

# Define features and target
X = data[['Close', 'Candlestick Shape', 'Candlestick Pattern', '200 MA', 'Primary Trend', 'RSI', 'Direction']]
y = data['Target']

# Encode categorical features
X = pd.get_dummies(X, columns=['Candlestick Shape', 'Candlestick Pattern', 'Primary Trend'])

# Scale numerical features
scaler = StandardScaler()
X[['Close', '200 MA', 'RSI']] = scaler.fit_transform(X[['Close', '200 MA', 'RSI']])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions on testing set
y_pred = gb_model.predict(X_test)

# Get the features for the last day in your dataset
last_day_features = X.iloc[-1].values.reshape(1, -1)

# Predict the target for tomorrow
predicted_target = gb_model.predict(last_day_features)[0]

# Print the predicted target
print('The predicted target for tomorrow is:', predicted_target)

# Add predictions and true target values to the testing set
X_test['Prediction'] = y_pred
X_test['True_Target'] = y_test

# Export testing set with predictions and true target values to a CSV file
X_test.to_csv('gb_predictions_with_true_targets.csv', index=False)

# Print classification report
print(classification_report(y_test, y_pred))
