import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load the data
data = pd.read_csv('pltr_with_features.csv')

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(data, data['Target'], test_size=0.2, random_state=42)

# Create a random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = rf_model.predict(X_test)

# Print the accuracy score
print('Accuracy:', accuracy_score(y_test, y_pred))
import joblib
# Save the model
joblib.dump(rf_model, 'rf_model.pkl')
