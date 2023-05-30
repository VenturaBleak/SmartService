import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('data_asset_by_postcode.csv', header=[0, 1])

# Flatten the multi-index header
df.columns = [' '.join(col).strip() for col in df.columns.values]

# Rename the first column to 'Postcode'
df.rename(columns={df.columns[0]: 'Postcode'}, inplace=True)

# Convert 'Postcode' to string to ensure it's not included in the numeric operations
df['Postcode'] = df['Postcode'].astype(str)

# Convert columns to numeric (excluding 'Postcode')
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill rows with NA with a 0
df = df.fillna(0)

# print all the rows where there are "0" values
zero_rows = df[(df == 0).any(axis=1)]
print(zero_rows)

## the rows with values of 0 are industrial areas which is why there are no assets

# create a new dataframe without the rows with "0" values
df = df[(df != 0).all(1)]

# print the distribution of each column
# iterate over each column
for column in df.columns[1:]:
    # plot the histogram and kernel density estimate
    sns.displot(df, x=column, kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Inferential statistics, including normality tests like the Shapiro-Wilk test, are used to make inferences about a
# population based on a sample from that population. They're a tool we use when we don't have access to every member
# of the population, which is often the case.
# Because this is the entire population, can be describe the distribution of that population directly. Directly compute
# and plot the distribution of data without needing to infer its shape.

# Calculate Z-scores (excluding 'Postcode')
columns_to_check = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(stats.zscore(df[columns_to_check]))

# Define a threshold
threshold = 3
outliers = np.where(z_scores > threshold)

#print all the outliers with their index, value, column and postcode
outlier_rows, outlier_cols = outliers
for row, col in zip(outlier_rows, outlier_cols):
    print(f"Outlier found at index {row}, Postcode {df.iloc[row]['Postcode']}, Column {columns_to_check[col]}, Value {df.iloc[row][columns_to_check[col]]}")

outlier_rows, outlier_cols = outliers
outlier_data = []

for row, col in zip(outlier_rows, outlier_cols):
    outlier_data.append([row, df.iloc[row]['Postcode'], columns_to_check[col], df.iloc[row][columns_to_check[col]]])

outlier_assets_df = pd.DataFrame(outlier_data, columns=['Index', 'Postcode', 'Column', 'Value'])

# save outliers_df in a csv file
outlier_assets_df.to_csv('outliers_assets.csv', index=False)

# Visualization (only for the first 3 data columns)
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[1]], label=df.columns[1])
plt.plot(df['Postcode'], df[df.columns[2]], label=df.columns[2])
plt.plot(df['Postcode'], df[df.columns[3]], label=df.columns[3])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[2]], color='r')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[3]], color='r')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Assets')  # Add y-axis label
plt.title('Anomaly detection')
plt.xticks(rotation=90)
plt.show()

# visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[4]], label=df.columns[4])
plt.plot(df['Postcode'], df[df.columns[5]], label=df.columns[5])
plt.plot(df['Postcode'], df[df.columns[6]], label=df.columns[6])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[4]], color='r', label='Outliers')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[5]], color='r')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[6]], color='r')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Assets')  # Add y-axis label
plt.title('Anomaly detection')
plt.xticks(rotation=90)
plt.show()

# visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[7]], label=df.columns[7])
plt.plot(df['Postcode'], df[df.columns[8]], label=df.columns[8])
plt.plot(df['Postcode'], df[df.columns[9]], label=df.columns[9])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[7]], color='r', label='Outliers')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[8]], color='r')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[9]], color='r')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Assets')  # Add y-axis label
plt.title('Anomaly detection')
plt.xticks(rotation=90)
plt.show()

# visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[10]], label=df.columns[10])
plt.plot(df['Postcode'], df[df.columns[11]], label=df.columns[11])
plt.plot(df['Postcode'], df[df.columns[12]], label=df.columns[12])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[10]], color='r', label='Outliers')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[11]], color='r')
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[12]], color='r')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Assets')  # Add y-axis label
plt.title('Anomaly detection')
plt.xticks(rotation=90)
plt.show()

# visualization for all columns
plt.figure(figsize=(10, 6))
for col in df.columns[1:]:
    plt.plot(df['Postcode'], df[col], label=col)
    plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][col], color='r')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Assets')  # Add y-axis label
plt.title('Anomaly detection')
plt.xticks(rotation=90)
plt.show()

# I would leave the outliers in the data set because they are not errors. They are just extreme values.
# The 1026ish postcodes seem to be relatively new districts that are built on artifical islands and seem to be quiet rich
# The 1105 neighborhoods seem to be mixed neighborhodds with also some poorer neighborhoods (they saw some restructuring in recent years though)
# The 1028 postcode is very remote and has a lot of nature and is therefore very expensive and this would explain the high assets (farmers with lots of land)
