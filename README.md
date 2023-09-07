# Apply-Gradient-Boosting-to-predict-stock-or-cryptocurrency-price
## README for Gradient Boosting Model Tuning with Data Preprocessing

---

### Description

This repository contains Python scripts for preprocessing financial data with technical indicators and tuning a Gradient Boosting Classifier. The preprocessing script loads data, adds various technical indicators, and saves the processed data. The model tuning script loads the processed data, preprocesses it further (including encoding categorical features and scaling numerical ones), splits the data into training and testing sets, and then creates a Gradient Boosting model.

### Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- TA-Lib (for technical analysis indicators)

### How to Run

1. Ensure you have all the required libraries installed.
2. For data preprocessing:
   - Place your dataset in the same directory as the preprocessing script. The script expects a file named 'eth_usd.csv'.
   - Run the preprocessing script using:

```bash
python preprocessing_data.py
```

3. For model tuning:
   - Ensure that the processed dataset is in the same directory as the tuning script. The script expects a file named 'eth_with_features.csv' or any name you have save with the file preprocessing_data.py .
   - Run the tuning script using:

```bash
python GradientBoosting_model_tuning.py
```

### File Structure

- `preprocessing_data.py`: Script for preprocessing data by adding technical indicators.
- `GradientBoosting_model_tuning.py`: Main script containing the model tuning code.

### Features and Target

For the model tuning:

- Features:
    - Close
    - Candlestick Shape
    - Candlestick Pattern
    - 200 MA
    - Primary Trend
    - RSI
    - Direction
- Target: 
    - Target

### Contributions

Feel free to fork this repository and contribute. Pull requests are welcome.

---


