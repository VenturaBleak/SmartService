import pandas as pd

# Read the CSV file
df = pd.read_csv('hybird_ams_passenger.csv')

# Remove empty rows because the first few rows are not included zip codes in other data about demogrpahics
df.dropna(inplace=True)

# Convert the Postcode column to string
df['Postcode'] = df['Postcode'].astype(str)

# drop all columns that are not postcode, December 2017, December 2018, December 2019, December 2020, December 2021, December 2022
df = df[['Postcode', 'December 2017', 'December 2018', 'December 2019', 'December 2020', 'December 2021', 'December 2022']]
print(df)

# drop the december in the column names
df.columns = ['Postcode', '2017', '2018', '2019', '2020', '2021', '2022']
print(df)

# Basic summary statistics
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Basic summary statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# export dataframe to csv
df.to_csv('ev_hybrid_ams_passenger_clean.csv', index=False)
