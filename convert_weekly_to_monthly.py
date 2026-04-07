import pandas as pd
import numpy as np

# Read the weekly data
df = pd.read_csv('data/fix/data_2025_ndvi.csv')

# Create a date column from Year and Week
# We'll use the first day of each week as the reference date
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2) + '-1', 
                             format='%Y-W%W-%w')

# Extract Year-Month for grouping
df['Year_Month'] = df['Date'].dt.to_period('M')

# Define columns to sum (dengue cases)
sum_columns = ['Cases']

# Define columns to average (environmental variables)
avg_columns = ['NDVI', 'Cloud_Cover', 'Humidity', 'Precipitation_Total', 
               'Temperature_Min', 'Temperature_Max', 'Temperature_Avg', 
               'Pressure', 'Wind_Speed', 'Wind_Direction']

# Define columns to keep as-is (identifiers)
keep_columns = ['Year', 'Region', 'Latitude', 'Longitude']

# Group by Region and Year_Month
grouped = df.groupby(['Region', 'Year_Month'])

# Aggregate data
monthly_data = grouped.agg({
    'Year': 'first',  # Keep the year
    'Cases': 'sum',  # Sum dengue cases
    'Latitude': 'first',  # Keep the same
    'Longitude': 'first',  # Keep the same
    'NDVI': 'mean',
    'Cloud_Cover': 'mean',
    'Humidity': 'mean',
    'Precipitation_Total': 'mean',
    'Temperature_Min': 'mean',
    'Temperature_Max': 'mean',
    'Temperature_Avg': 'mean',
    'Pressure': 'mean',
    'Wind_Speed': 'mean',
    'Wind_Direction': 'mean'
}).reset_index()

# Add a Month column (1-12)
monthly_data['Month'] = monthly_data['Year_Month'].dt.month

# Reorder columns to match original structure (with Month instead of Week)
column_order = ['Year', 'Region', 'Month', 'Cases', 'Latitude', 'Longitude',
                'NDVI', 'Cloud_Cover', 'Humidity', 'Precipitation_Total',
                'Temperature_Min', 'Temperature_Max', 'Temperature_Avg',
                'Pressure', 'Wind_Speed', 'Wind_Direction']

monthly_data = monthly_data[column_order]

# Sort by Region, Year, and Month
monthly_data = monthly_data.sort_values(['Region', 'Year', 'Month']).reset_index(drop=True)

# Save to CSV
monthly_data.to_csv('data/fix/data_monthly_5kab_2025_ndvi.csv', index=False)

print(f"Conversion complete!")
print(f"Original weekly data shape: {df.shape}")
print(f"Monthly data shape: {monthly_data.shape}")
print(f"\nFirst few rows of monthly data:")
print(monthly_data.head(15))
print(f"\nSample comparison for one region:")
print(f"Weekly data (first month):")
sample_region = df['Region'].iloc[0]
sample_weekly = df[(df['Region'] == sample_region) & (df['Year'] == 2021) & (df['Week'] <= 4)]
print(sample_weekly[['Year', 'Week', 'Cases', 'Temperature_Avg', 'Humidity']])
print(f"\nMonthly data (first month):")
sample_monthly = monthly_data[(monthly_data['Region'] == sample_region) & (monthly_data['Year'] == 2021) & (monthly_data['Month'] == 1)]
print(sample_monthly[['Year', 'Month', 'Cases', 'Temperature_Avg', 'Humidity']])
