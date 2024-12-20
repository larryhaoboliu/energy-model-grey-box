import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# Load data
## Folder Path, 
folder_path = "/Users/haobo2022/Documents/energy-model-black-box/energy usage data"
pathlist = Path(folder_path).glob('*.csv')
comb_data = pd.DataFrame()


# Reindex data for analysis and plotting
for path in pathlist:
    try:
        raw_data = pd.read_csv(str(path))
        reindex_data = raw_data.reset_index()
        reindex_data.columns = ["Account ID", "Service Address", "Date", "Consumption (kWh)", "Temperature(F)"]
        data_clean = reindex_data.drop(0, axis=0).copy()
        data_final = data_clean[["Date", "Consumption (kWh)", "Temperature(F)"]].copy()
        year = ' ' + str(path)[-8:-4]
        data_final.loc[:, 'Date'] = pd.to_datetime(data_final['Date'] + year, format='%b %d %Y')
        comb_data = pd.concat([comb_data, data_final], ignore_index=True)
    except Exception as e:
        print(f"Error processing {path}: {e}")

comb_data = comb_data.sort_values(by="Date").reset_index(drop=True)

# Interpolate and clean data
z_thresh = 3
comb_data['Temperature(F)'].replace([0, np.inf, -np.inf], pd.NA, inplace=True)
comb_data['Consumption (kWh)'].replace([0, np.inf, -np.inf], pd.NA, inplace=True)
comb_data['Temperature(F)'] = pd.to_numeric(comb_data['Temperature(F)'], errors='coerce')
comb_data['Consumption (kWh)'] = pd.to_numeric(comb_data['Consumption (kWh)'], errors='coerce')

mean_temp = comb_data['Temperature(F)'].mean()
std_temp = comb_data['Temperature(F)'].std()
mean_cons = comb_data['Consumption (kWh)'].mean()
std_cons = comb_data['Consumption (kWh)'].std()

comb_data.loc[((comb_data['Temperature(F)'] - mean_temp) / std_temp).abs() > z_thresh, 'Temperature(F)'] = pd.NA
comb_data.loc[((comb_data['Consumption (kWh)'] - mean_cons) / std_cons).abs() > z_thresh, 'Consumption (kWh)'] = pd.NA

comb_data['Temperature(F)'] = comb_data['Temperature(F)'].interpolate(method='linear').ffill().bfill()
comb_data['Consumption (kWh)'] = comb_data['Consumption (kWh)'].interpolate(method='linear').ffill().bfill()

# Plot data
comb_data.replace([np.inf, -np.inf], np.nan, inplace=True)
comb_data.dropna(subset=['Temperature(F)', 'Consumption (kWh)', 'Day'], inplace=True)

fig = plt.figure(figsize=(12, 8), dpi=250)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

ax1 = fig.add_subplot(gs[0])
sns.set_style("whitegrid")
sns.lineplot(x='Day', y='Temperature(F)', data=comb_data, ax=ax1, color='red', linewidth=2, linestyle='--')
ax1.set_title('Daily Temperature Changes Over Time', fontsize=16)
ax1.set_ylabel('Temperature (F)', fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45)

ax2 = fig.add_subplot(gs[1], sharex=ax1)
sns.lineplot(x='Day', y='Consumption (kWh)', data=comb_data, ax=ax2, color='blue', linewidth=2)
ax2.set_title('Daily Energy Consumption Over Time', fontsize=16)
ax2.set_ylabel('Energy Consumption (kWh)', fontsize=12)
plt.setp(ax2.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# Train and evaluate model
comb_data['Day'] = pd.to_datetime(comb_data['Day'], errors='coerce')
comb_data['DayOfWeek'] = comb_data['Day'].dt.dayofweek
comb_data['Month'] = comb_data['Day'].dt.month
comb_data['IsWeekend'] = comb_data['DayOfWeek'].apply(lambda x: x >= 5)

X = comb_data[['Temperature(F)', 'DayOfWeek', 'Month', 'IsWeekend']]
y = comb_data['Consumption (kWh)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

plt.figure(figsize=(10, 6), dpi=250)
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Time Index')
plt.ylabel('Consumption (kWh)')
plt.grid(True)
plt.show()