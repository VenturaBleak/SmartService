import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Load the data from CSV
df = pd.read_csv('data_land_use_by_postcode.csv')

# Convert columns to numeric
numeric_columns = df.columns[4:]  # Exclude non-numeric columns
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Postcode' column to string
df['Postcode'] = df['Postcode'].astype(str)

# Basic summary statistics
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Basic summary statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# Identify anomalies using box plots
plt.figure(figsize=(10, 6))
df.boxplot(column=['traffic_area', 'built_up_area', 'semi_built_up_area',
                   'recreational_area', 'agrarian_area', 'forest_nature_are',
                   'water_body_area', 'total_area'])
plt.title('Land Use Distribution')
plt.ylabel('Area')
plt.show()

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df[numeric_columns]))

# Define a threshold
threshold = 3
outliers = np.where(z_scores > threshold)

# Print outliers
print(f"Outliers are at index positions: {outliers}")

# Visualization
plt.figure(figsize=(10, 6))
for col in numeric_columns:
    plt.plot(df['Postcode'], df[col], label=col)

plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][numeric_columns[0]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')
plt.ylabel('Area')
plt.title('Area Distribution')
plt.show()

# Identify the rows with outliers
outlier_rows = df.loc[outliers[0]]

# Drop duplicate rows to display each row only once
outlier_rows = outlier_rows.drop_duplicates()

pd.set_option('display.max_rows', None)

# Print the outlier rows
print(outlier_rows)

# do the same analysis as above but with grouping the variable by postcode of the first 4 digits

# Extract the first four digits from the Postcode column and remove non-numeric characters
df['Postcode_group'] = df['Postcode'].str.extract(r'(\d{4})').astype(float)

# print the first 5 rows for the new column
print(df.head())

# Select only the numeric columns
numeric_columns = df.columns[4:]  # Assuming the first three columns are strings

# Convert the numeric columns to numeric data type
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by 'Postcode_group' and calculate the mean of numeric columns
df_grouped = df.groupby('Postcode_group')[numeric_columns].mean()

# Print the resulting grouped dataframe
print(df_grouped)

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df_grouped[numeric_columns]))

# Define a threshold
threshold = 3
outliers = np.where(z_scores > threshold)

# Print outliers
print(f"Outliers are at index positions: {outliers}")

# Visualization
plt.figure(figsize=(12, 6))
for col in numeric_columns:
    plt.plot(range(len(df_grouped.index)), df_grouped[col], label=col)

plt.scatter(outliers, df_grouped.iloc[outliers][numeric_columns[0]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode Group')
plt.ylabel('Area')
plt.title('Area Distribution')

# Set x-axis tick labels to show all postcode groups
plt.xticks(range(len(df_grouped.index)), df_grouped.index, rotation=90)

plt.show()

# Identify the rows with outliers
outlier_rows = df.loc[outliers[0]]

# Drop duplicate rows to display each row only once
outlier_rows = outlier_rows.drop_duplicates()

# Print the outlier rows
print(outlier_rows)

# Define a threshold
threshold = 3
outliers_row_col_indices = np.where(z_scores > threshold)

# Create a DataFrame for outliers
outliers_df = pd.DataFrame(columns=df.columns)
for i in range(len(outliers_row_col_indices[0])):
    row_index = outliers_row_col_indices[0][i]
    col_index = outliers_row_col_indices[1][i]
    outliers_df = pd.concat([outliers_df, df.iloc[[row_index], :]])


# save outliers_df in a csv file
outliers_df.to_csv('outliers_land_use.csv', index=False)


## I would probably exclude some of the outliers bc some areas are harbours etc. and not really land or useful for the model?
