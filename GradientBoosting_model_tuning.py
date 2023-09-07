import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('pltr_with_features.csv')

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

# Create Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)

# Define hyperparameter search space
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(gb_model, param_grid, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
print('Best hyperparameters:', grid_search.best_params_)

# Use the best model for predictions
best_model = grid_search.best_estimator_

# Make predictions on testing set
y_pred = best_model.predict(X_test)

# Get the features for the last day in your dataset
last_day_features = X.iloc[-1].values.reshape(1, -1)

# Predict the target for tomorrow
predicted_target = best_model.predict(last_day_features)[0]

# Print the predicted target
print('The predicted target for tomorrow is:', predicted_target)

# Add predictions and true target values to the testing set
X_test['Prediction'] = y_pred
X_test['True_Target'] = y_test

# Export testing set with predictions and true target values to a CSV file
X_test.to_csv('gb_predictions_with_true_targets_tuned.csv', index=False)


# Print classification report
print(classification_report(y_test, y_pred))

import joblib

# Save the best model
joblib.dump(best_model, 'best_gb_model.pkl')
