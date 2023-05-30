import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import shapiro

# Read the CSV file (which I cleaned beforehand to only include the last month of each year)
df = pd.read_csv('ev_ams_passenger_clean.csv')

# Extract the postcodes and years
postcodes = df['Postcode']
years = df.columns[1:]

# test for a normal distribution
for year in years:
    stat, p = shapiro(df[year])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print(f'{year} looks Gaussian (fail to reject H0)')
    else:
        print(f'{year} does not look Gaussian (reject H0)')


# Perform anomaly detection for each year
for year in years:
    # Create a new figure for each year
    plt.figure(figsize=(10, 6))

    # Prepare the data for anomaly detection
    X = df[[year]]
    X.columns = ['Number of Passenger Cars']  # Set valid feature names

    # Fit the Isolation Forest model
    model = IsolationForest(contamination=0.05)
    model.fit(X)

    # Predict the anomalies
    anomalies = model.predict(X)

    # Plot the number of passenger cars per postcode
    plt.scatter(postcodes, df[year], label='Number of Cars')

    # Highlight the anomalies with a thicker line
    plt.scatter(postcodes[anomalies == -1], df[year][anomalies == -1], color='red', label='Anomalies', linewidths=2)

    # Set labels and title
    plt.xlabel('Postcode')
    plt.ylabel('Number of Passenger Cars')
    plt.title(f'Anomaly Detection - Year {year}')

    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=90)
    plt.gca().set_xticks(postcodes)  # Set all postcodes as x-axis ticks

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()


# Create a new figure
plt.figure(figsize=(10, 6))

# Plot the trends in the number of passenger cars per postcode for each year
for year in df.columns[1:]:
    # Extract the year from the column name
    year_value = year.split('_')[0]

    # Get the data for the current year
    data = df[year].tolist()

    if year == 'Anomaly Score':
        # Plot the anomaly score line with a thicker and more outstanding style
        plt.plot(postcodes, data, label=year_value, linewidth=2, linestyle='--')
    else:
        # Plot the regular trend lines
        plt.plot(postcodes, data, label=year_value)

# Set labels and title
plt.xlabel('Postcode')
plt.ylabel('Number of Passenger Cars')
plt.title('Trends in Number of Passenger Cars per Postcode')

# Customize the x-axis tick labels
plt.xticks(rotation=90)
plt.gca().set_xticks(postcodes)  # Set all postcodes as x-axis ticks

# Show the legend
plt.legend()

# Show the plot
plt.show()

# show me all the outliers including the value out the number of cars



for year in years:
    # Prepare the data for anomaly detection
    X = df[[year]]
    X.columns = ['Number of Passenger Cars']  # Set valid feature names

    # Fit the Isolation Forest model
    model = IsolationForest(contamination=0.05)
    model.fit(X)

    # Predict the anomalies
    anomalies = model.predict(X)

    # Print the outliers
    print(f'Outliers for year {year}:')
    outliers_df = df[anomalies == -1][['Postcode', year]]
    print(outliers_df)
    print()

# these are the postcodes that have the most outliers: 1013, 1033, 1083, 1102, 1082, 1097, 1101
# In zipcode 1097 a carsharing company is registered, so that might explain the high number of cars
# Zipcode 1101 is an industrial area, so that might explain the high number of cars

# 1013: This is part of the Centrum borough, which is the city center of Amsterdam. This area includes parts of the Jordaan
# neighborhood, known for its beautiful canals, and historic buildings.
# 1033: This is in the Amsterdam Noord (North) borough. This area has a mix of residential areas and businesses, including
# some large industrial zones.
# 1083: This postcode is located in the Buitenveldert neighborhood of the Zuid (South) borough. Buitenveldert is a quieter,
# residential area, located near the Amsterdamse Bos (Amsterdam Forest) and the Zuidas business district.
# 1102: This is part of the Bijlmermeer neighborhood, generally just called "Bijlmer," located in the Zuidoost (Southeast)
# borough. The Bijlmer is known for its high-rise apartment buildings and is one of the most multicultural areas of Amsterdam.
# 1082: This area is also in the Zuid (South) borough, covering part of the Buitenveldert neighborhood and the Zuidas
# business district. Zuidas is known as Amsterdam's "second city center" and is home to many corporate headquarters, law firms, and financial institutions.
# 1097: This postcode covers part of the Watergraafsmeer area in the Oost (East) borough. Watergraafsmeer is a residential
# neighborhood known for its spacious layout and green spaces.
# 1101: This is another area in the Bijlmermeer neighborhood in the Zuidoost borough. It also includes parts of the ArenAPoort area,
# where you'll find the Johan Cruijff ArenA (the stadium where Ajax football club plays), AFAS Live, and the Amsterdamse Poort shopping center.


