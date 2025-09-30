# Ex.No: 6               HOLT WINTERS METHOD
### Date: 30-09-2025

#### NAME: JERUSHLIN JOSE JB
#### REGISTER NUMBER:212222240039

### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# Load the dataset
file_path = "NFLX.csv"  # Update with your actual path if needed
data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

# Use only the Close (Netflix stock price) column
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])

# Resample the data to monthly frequency (mean of each month)
monthly_data = data['Close'].resample('MS').mean()

# Split the data into train and test sets (90% train, 10% test)
train_data = monthly_data[:int(0.9 * len(monthly_data))]
test_data = monthly_data[int(0.9 * len(monthly_data)):]

# Holt-Winters model with additive trend and seasonality
fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12  # yearly seasonality for monthly data
).fit()

# Forecast on the test set
test_predictions = fitted_model.forecast(len(test_data))

# Plot the results
plt.figure(figsize=(12, 8))
train_data.plot(legend=True, label='Train')
test_data.plot(legend=True, label='Test')
test_predictions.plot(legend=True, label='Predicted')
plt.title('Train, Test, and Predicted Netflix Prices (Holt-Winters)')
plt.show()

# Evaluate model performance
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.4f}")
print(f"Mean Squared Error = {mse:.4f}")

# Fit the model to the entire dataset and forecast the future
final_model = ExponentialSmoothing(
    monthly_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

forecast_predictions = final_model.forecast(steps=12)  # Forecast 12 future months

# Plot the original and forecasted data
plt.figure(figsize=(12, 8))
monthly_data.plot(legend=True, label='Original Data')
forecast_predictions.plot(legend=True, label='Forecasted Data', color='red')
plt.title('Original and Forecasted Netflix Stock Prices (Holt-Winters Additive)')
plt.show()
```
### OUTPUT:

<img width="986" height="700" alt="image" src="https://github.com/user-attachments/assets/9dfb457b-53b0-4c5e-b531-d348f08f4923" />
<img width="988" height="735" alt="image" src="https://github.com/user-attachments/assets/4d6dbefe-2c53-425c-8936-35ef19a56435" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
