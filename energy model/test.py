import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec

pathlist = Path("/Users/haobo2022/Documents/energy-model-black-box/energy usage data").glob('*.csv')
temp_path = ""
comb_data = pd.DataFrame()

for path in pathlist:
    raw_data = pd.read_csv(str(path))
    reindex_data = raw_data.reset_index()
    reindex_data.columns = "Account ID", "Service Address", "Day", "Consumption (kWh)", "Temperature(F)"
    data_clean = reindex_data.drop(0, axis = 0)
    data_final = data_clean[["Day", "Consumption (kWh)", "Temperature(F)"]]
    year = ' ' +  str(path)[-8:-4]
    data_final['Day'] = pd.to_datetime(data_final['Day'] + str(year), format='%b %d %Y')
    comb_data = pd.concat([comb_data, data_final], ignore_index=True)


data = comb_data.sort_values(by = "Day").reset_index(drop = True)

## interpolation method

# Replace 0 values with NaN
data['Temperature(F)'].replace(0, pd.NA, inplace=True)

# Interpolate NaN values (linear interpolation by default)
data['Temperature(F)'] = data['Temperature(F)'].interpolate()

# Optionally, if the first or last value is NaN after interpolation, you can fill it:
# Fill any remaining NaN values with forward fill or backward fill
data['Temperature(F)'].fillna(method='ffill', inplace=True)  # or 'bfill'


# ### mean method
# # Calculate the mean temperature, excluding 0 values
# mean_temp = pd.to_numeric(data[data['Temperature(F)'] != 0]['Temperature(F)']).mean()

# # Replace 0 values with the mean temperature
# data['Temperature(F)'].replace(0, mean_temp, inplace=True)


# Convert 'Consumption (kWh)' and 'Temperature(F)' to numeric
data['Consumption (kWh)'] = pd.to_numeric(data['Consumption (kWh)'], errors='coerce')
data['Temperature(F)'] = pd.to_numeric(data['Temperature(F)'], errors='coerce')

# Set up a grid with different heights for the two subplots
fig = plt.figure(figsize=(12, 8), dpi = 250)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])  # Temperature plot shorter than energy plot

# First subplot (top): Temperature
ax1 = fig.add_subplot(gs[0])
sns.set_style("whitegrid")
sns.lineplot(x='Day', y='Temperature(F)', data=data, ax=ax1, color='red', linewidth=2, linestyle='--')
ax1.set_title('Daily Temperature Changes Over Time', fontsize=16)
ax1.set_ylabel('Temperature (F)', fontsize=12)

# Remove all vertical gridlines and keep only horizontal gridlines
ax1.grid(True, which='major', axis='y', color='gray', linestyle='-', linewidth=0.7)
ax1.grid(False, which='major', axis='x')

# Adjust x-axis to show more date labels
ax1.xaxis.set_major_locator(plt.MaxNLocator(15))
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.setp(ax1.get_xticklabels(), rotation=45)

# Second subplot (bottom): Energy Consumption
ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Share the same x-axis as the first plot
sns.lineplot(x='Day', y='Consumption (kWh)', data=data, ax=ax2, color='blue', linewidth=2)
ax2.set_title('Daily Energy Consumption Over Time', fontsize=16)
ax2.set_ylabel('Energy Consumption (kWh)', fontsize=12)

# Adjust x-axis to show more date labels and avoid white space
ax2.xaxis.set_major_locator(plt.MaxNLocator(15))
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.setp(ax2.get_xticklabels(), rotation=45)

# Remove vertical gridlines and keep only horizontal gridlines
ax2.grid(True, which='major', axis='y', color='gray', linestyle='-', linewidth=0.7)
ax2.grid(False, which='major', axis='x')

# Adjust layout to ensure no overlap between the subplots
plt.tight_layout()

# Show theplot
plt.show()


# exploring data correlation
correlation = data[['Consumption (kWh)', 'Temperature(F)']].corr()
print(correlation)


data['DayOfWeek'] = data['Day'].dt.dayofweek  # Monday=0, Sunday=6
data['Month'] = data['Day'].dt.month
data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: True if x >= 5 else False)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define features (X) and target variable (y)
X = data[['Temperature(F)', 'DayOfWeek', 'Month', 'IsWeekend']]  # Add other features as needed
y = data['Consumption (kWh)']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Instantiate the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6), dpi = 250)
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel()
plt.show()


