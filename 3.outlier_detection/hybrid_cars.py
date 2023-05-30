import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import shapiro

# Read the CSV file (which I cleaned beforehand to only inlcude the last month of each year)
df = pd.read_csv('ev_hybrid_ams_passenger_clean.csv')

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

# these are the postcodes that have the most outliers: 1033, 1102, 1101, 1043, 1077, 1071, 1012
# 1033: This postcode falls in the North borough, which is mainly a residential and industrial area. It has seen a lot of
# development in recent years, with new residential and cultural hotspots popping up.
# 1102: This postcode belongs to Amsterdam Zuidoost, specifically to the Bijlmermeer neighborhood, which is one of the city's
# most diverse and vibrant areas with a strong sense of community and an array of international food and music festivals.
# 1101: This postcode is also part of the Bijlmermeer neighborhood. It includes the ArenAPoort area, with the Johan Cruijff ArenA,
# the Ziggo Dome, and the large Amsterdamse Poort shopping center.
# 1043: This postcode falls within Sloterdijk, part of the West borough. This area is home to the bustling Sloterdijk
# Station, one of Amsterdam's major transport hubs, and has numerous office buildings. There are ongoing urban development
# projects to add more residential and leisure facilities to this area.
# 1077: This postcode is located in Amsterdam Zuid, and it includes part of the upscale Oud Zuid neighborhood and the
# southern part of the bustling De Pijp neighborhood. It is a culturally diverse area known for its cafés, restaurants,
# boutiques, and the famous Albert Cuyp Market.
# 1071: This is also part of the Oud Zuid neighborhood. This is one of the most prestigious and sought-after areas in
# the city, known for its historic architecture, the Van Gogh Museum, the Rijksmuseum, and the upscale shopping street,
# P.C. Hooftstraat. The famous Vondelpark also falls within this postcode.
# 1012: This postcode includes much of the city's historic heart, the Centrum borough. It includes the Red Light District,
# Dam Square with the Royal Palace, and many other popular tourist destinations.
