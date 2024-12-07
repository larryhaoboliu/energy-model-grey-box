## combined file of all functions and workflow
## import library (don't forget to download all of libraries ahead of time)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

## Loads and cleans energy usage data from multiple CSV files in the specified folder path. 
## Combines all data into a single DataFrame.
def load_and_clean_data(folder_path):
    pathlist = Path(folder_path).glob('*.csv')
    comb_data = pd.DataFrame()
    
    for path in pathlist:
        try:
            raw_data = pd.read_csv(str(path)) 
            reindex_data = raw_data.reset_index()
            reindex_data.columns = ["Account ID", "Service Address", "Day", "Consumption (kWh)", "Temperature(F)"]
            data_clean = reindex_data.drop(0, axis=0)  
            data_final = data_clean[["Day", "Consumption (kWh)", "Temperature(F)"]] 
            year = ' ' + str(path)[-8:-4] 
            data_final['Day'] = pd.to_datetime(data_final['Day'] + year, format='%b %d %Y')
            comb_data = pd.concat([comb_data, data_final], ignore_index=True)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    comb_data = comb_data.sort_values(by="Day").reset_index(drop=True)
    return comb_data

## try using multi-day mean to accomodate for missing temperature and consumption data.
def interpolate_data(data):
    data['Temperature(F)'].replace(0, pd.NA, inplace=True)
    data['Consumption (kWh)'].replace(0, pd.NA, inplace=True)

    # Define mean interpolation function
    def mean_interpolation(series):
        for i in range(len(series)):
            if pd.isna(series.iloc[i]):  
                start_idx = max(i - 3, 0) 
                end_idx = min(i + 3 + 1, len(series)) 
                mean_value = series.iloc[start_idx:end_idx].dropna().mean() 
                series.iloc[i] = mean_value 
        return series

    # Apply the mean interpolation to both columns
    data['Temperature(F)'] = mean_interpolation(data['Temperature(F)'])
    data['Consumption (kWh)'] = mean_interpolation(data['Consumption (kWh)'])

    # Fill any remaining NaN values using forward fill or backward fill
    data['Temperature(F)'].fillna(method='ffill', inplace=True)
    data['Temperature(F)'].fillna(method='bfill', inplace=True)
    data['Consumption (kWh)'].fillna(method='ffill', inplace=True)
    data['Consumption (kWh)'].fillna(method='bfill', inplace=True)

    # Ensure numeric conversion
    data['Temperature(F)'] = pd.to_numeric(data['Temperature(F)'], errors='coerce')
    data['Consumption (kWh)'] = pd.to_numeric(data['Consumption (kWh)'], errors='coerce')

    return data

## Creates line plots for daily temperature changes and energy consumption over time.

def plot_data(data):
    fig = plt.figure(figsize=(12, 8), dpi=250) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
    
    # Plot daily temperature
    ax1 = fig.add_subplot(gs[0])
    sns.set_style("whitegrid")
    sns.lineplot(x='Day', y='Temperature(F)', data=data, ax=ax1, color='red', linewidth=2, linestyle='--')
    ax1.set_title('Daily Temperature Changes Over Time', fontsize=16)
    ax1.set_ylabel('Temperature (F)', fontsize=12)
    ax1.grid(True, which='major', axis='y', color='gray', linestyle='-', linewidth=0.7)
    ax1.grid(False, which='major', axis='x')  
    ax1.xaxis.set_major_locator(plt.MaxNLocator(15)) 
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45) 
    
    # Plot daily energy consumption
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  
    sns.lineplot(x='Day', y='Consumption (kWh)', data=data, ax=ax2, color='blue', linewidth=2)
    ax2.set_title('Daily Energy Consumption Over Time', fontsize=16)
    ax2.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=45)
    ax2.grid(True, which='major', axis='y', color='gray', linestyle='-', linewidth=0.7)
    ax2.grid(False, which='major', axis='x') 
    
    plt.tight_layout() 
    plt.show()


# Trains a Random Forest Regressor model to predict energy consumption and evaluates its performance.
def train_and_evaluate_model(data):
    # Day catagorization
    data['DayOfWeek'] = data['Day'].dt.dayofweek 
    data['Month'] = data['Day'].dt.month  
    data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: x >= 5) 
    
    # Defining variables
    X = data[['Temperature(F)', 'DayOfWeek', 'Month', 'IsWeekend']]
    y = data['Consumption (kWh)']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')  
    print(f'R-squared: {r2}') 
    
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 6), dpi=250)
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Energy Consumption')
    plt.xlabel('Time Index')
    plt.ylabel('Consumption (kWh)')
    plt.show()


## Import xgboost from library
from xgboost import XGBRegressor

## Trains an XGBoost Regressor model to predict energy consumption and evaluates its performance.
def train_and_evaluate_with_xgboost(data):

	# Day categorization
    data['DayOfWeek'] = data['Day'].dt.dayofweek
    data['Month'] = data['Day'].dt.month
    data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: x >= 5)
    
    # Defining variables
    X = data[['Temperature(F)', 'DayOfWeek', 'Month', 'IsWeekend']]
    y = data['Consumption (kWh)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost Regressor
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6), dpi=250)
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='-')
    plt.legend()
    plt.title('Actual vs Predicted Energy Consumption (XGBoost)')
    plt.xlabel('Time Index')
    plt.ylabel('Consumption (kWh)')
    plt.grid(True)
    plt.show()

# Workflow
folder_path = "/Users/haobo2022/Documents/energy-model-black-box/energy usage data"
data = load_and_clean_data(folder_path)  # Load and clean data
data = interpolate_data(data)  # Interpolate missing data
plot_data(data)  # Plot the cleaned data
train_and_evaluate_model(data)  # Train and evaluate the model
train_and_evaluate_with_xgboost(data)  # Train and evaluate with XGBoost