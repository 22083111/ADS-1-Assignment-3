import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.cluster as cl
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import t

def merge_indicators():
    """
    Merge gas and liquid indicators from two CSV files for a specific year.

    Returns:
    merged_data (pd.DataFrame): Merged data containing gas and liquid indicators for a specific year.
    """
    # Read the first CSV file into a DataFrame
    gas = pd.read_csv("gas.csv")

    # Read the second CSV file into another DataFrame
    liquid = pd.read_csv("liquid.csv")

    # Select a specific year (e.g., 1990)
    selected_year = '1990'

    # Extract relevant columns for the selected year from both DataFrames
    gas_selected = gas[['Country Name', selected_year]]
    liquid_selected = liquid[['Country Name', selected_year]]

    # Merge the two DataFrames on the 'Country Name' column
    merged_data = pd.merge(gas_selected, liquid_selected, on='Country Name', suffixes=('_gas', '_liquid'))

    # Drop rows with NaN values
    merged_data = merged_data.dropna()

    # Save the merged data to a new CSV file
    merged_data.to_csv(f'merged_indicators_{selected_year}.csv', index=False)

    # Rename the data column to "liquid"
    merged_data = merged_data.rename(columns={selected_year + '_liquid': 'liquid'})

    # Copy the "1990" column from the "gas" DataFrame to the "gas" column in the merged DataFrame
    merged_data["gas"] = gas[selected_year]

    return merged_data

def setup_kmeans_clusterer(data, features, n_clusters=4, n_init=20):
    """
    Set up a KMeans clusterer for the given data and features.

    Args:
    data (pd.DataFrame): Input data for clustering.
    features (list): List of feature columns for clustering.
    n_clusters (int): Number of clusters to form.
    n_init (int): Number of times the k-means algorithm will be run with different centroid seeds.

    Returns:
    clusters (KMeans): Trained KMeans clusterer.
    scaler (RobustScaler): Fitted RobustScaler for normalization.
    """
    # Normalize the data using a robust scaler
    scaler = pp.RobustScaler()
    to_clust = data[features]
    scaler.fit(to_clust)
    norm = scaler.transform(to_clust)

    # Setting up the clustering function
    clusters = cl.KMeans(n_clusters=n_clusters, n_init=n_init)

    # Doing the clustering. The result will be stored in clusters
    clusters.fit(norm)

    return clusters, scaler

def compare_countries_in_clusters(data, labels, country_column="Country Name", cluster_column="labels"):
    """
    Compare countries in different clusters based on a representative from each cluster.

    Args:
    data (pd.DataFrame): Input data containing country information and cluster labels.
    labels (array-like): Cluster labels assigned by a clustering algorithm.
    country_column (str): Column name containing country names.
    cluster_column (str): Column name containing cluster labels.
    """
    # Create a dictionary to store representatives from each cluster
    cluster_representatives = {}

    # Identify a representative country from each cluster
    for cluster_label in set(labels):
        cluster_data = data[data[cluster_column] == cluster_label]
        representative_country = cluster_data.iloc[0][country_column]
        cluster_representatives[cluster_label] = representative_country

    # Compare countries from different clusters
    for cluster_label_1, country_1 in cluster_representatives.items():
        for cluster_label_2, country_2 in cluster_representatives.items():
            if cluster_label_1 < cluster_label_2:
                compare_countries(data, country_1, country_2)

def compare_countries(data, country_1, country_2):
    """
    Compare two countries based on specified criteria.

    Args:
    data (pd.DataFrame): Input data containing relevant information.
    country_1 (str): Name of the first country to compare.
    country_2 (str): Name of the second country to compare.
    """
    # Print or perform any comparison between the specified countries
    print(f"Comparing {country_1} and {country_2}")

def main():
    """
    Main function to execute the code.
    """
    # Call the function to execute the code and get the merged data
    both = merge_indicators()

    # Define the features for clustering
    clustering_features = ["gas", "liquid"]

    # Set up the KMeans clusterer
    kmeans_clusterer, scaler = setup_kmeans_clusterer(both, clustering_features)

    # Extracting the labels, i.e., the cluster number
    labels = kmeans_clusterer.labels_

    # Extract the cluster centres and rescale them
    centres = kmeans_clusterer.cluster_centers_
    centres = scaler.inverse_transform(centres)

    # Extract x and y values from the rescaled cluster centres
    xcen = centres[:, 0]
    ycen = centres[:, 1]

    # Select a colour map with high contrast
    cm = matplotlib.cm.get_cmap("Paired")

    # Scatter plot for data points with different colors for each cluster
    plt.scatter(both["gas"], both["liquid"], s=10, c=labels, marker="o", cmap=cm)

    # Scatter plot for cluster centers with a black diamond marker
    plt.scatter(xcen, ycen, s=45, c='k', marker="D", label='Cluster Centers')

    # Scatter plot for each cluster separately with distinct markers
    for i in range(len(set(labels))):
        cluster_data = both[labels == i]
        plt.scatter(cluster_data["gas"], cluster_data["liquid"], s=15, marker="o", cmap=cm, label=f'Cluster {i}')

    plt.xlabel("CO2 emissions from gaseous fuel consumption (kt)")
    plt.ylabel("CO2 emissions from liquid fuel consumption (kt)")

    # Add a title to the graph
    plt.title("KMeans Clustering - CO2 Emmissions of Gaseous vs Liquid in 1990")

    # Add legend with cluster labels
    plt.legend()

    plt.show()

    # Adding the cluster membership information to the dataframe
    both["labels"] = labels

    # Write the dataframe into an Excel file
    both.to_excel("cluster_results.xlsx", index=False)

    # Compare countries in different clusters
    compare_countries_in_clusters(both, labels)

if __name__ == "__main__":
    main()



def polynomial_model(x, *coefficients):
    """
    Evaluate the polynomial model at given x values using the provided coefficients.

    Parameters:
    - x (array-like): Input values for the polynomial.
    - coefficients (float): Coefficients of the polynomial model.

    Returns:
    - array-like: Output values of the polynomial model.
    """
    return np.polyval(coefficients, x)

def fit_polynomial_model(x, y, degree):
    """
    Fit a polynomial model to the given data and return the coefficients.

    Parameters:
    - x (array-like): Independent variable values.
    - y (array-like): Dependent variable values.
    - degree (int): Degree of the polynomial model.

    Returns:
    - array-like: Coefficients of the fitted polynomial model.
    """
    coefficients = np.polyfit(x, y, degree)
    return coefficients

def err_ranges_polynomial(x, coefficients, y, confidence=0.95):
    """
    Calculate confidence intervals for a polynomial model.

    Parameters:
    - x (array-like): Independent variable values.
    - coefficients (array-like): Coefficients of the polynomial model.
    - y (array-like): Dependent variable values.
    - confidence (float): Confidence level for the interval.

    Returns:
    - tuple: Lower and upper bounds of the confidence interval.
    """
    p = np.poly1d(coefficients)
    y_fit = p(x)
    p_err = np.sqrt((1/(len(x)-len(coefficients))) * np.sum((y - y_fit)**2))
    delta = abs(t.ppf((1 - confidence) / 2., len(x)-len(coefficients))) * p_err
    upper = p(x) + delta
    lower = p(x) - delta
    return lower, upper

def create_plot(x_data, y_data, predicted_values, confidence_intervals, country_name, degree):
    """
    Create a plot with actual data, polynomial model, and confidence intervals.

    Parameters:
    - x_data (array-like): Independent variable values from the actual data.
    - y_data (array-like): Dependent variable values from the actual data.
    - predicted_values (array-like): Predicted values from the polynomial model.
    - confidence_intervals (tuple): Lower and upper bounds of the confidence interval.
    - country_name (str): Name of the country for plot title.
    - degree (int): Degree of the polynomial model.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 7))
    plt.scatter(x_data, y_data, color='blue', label='Actual Data')
    plt.plot(extended_years_range, predicted_values, label=f'Polynomial Model (Degree {degree})', color='green', linestyle='--')
    plt.fill_between(x_data, confidence_intervals[0], confidence_intervals[1], color='green', alpha=0.2, label='95% Confidence Interval')

    for year in [2020, 2022]:
        plt.plot(year, polynomial_model(year, *poly_coefficients_range_clean_adj), 'ro')
        plt.text(year, polynomial_model(year, *poly_coefficients_range_clean_adj), f'  Predicted {year}', verticalalignment='bottom')

    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions from Liquid Fuel (kt)')
    plt.title(f'CO2 Emissions from Liquid Fuel in {country_name}\nPolynomial Model and Predictions')
    plt.scatter([], [], color='red', marker='o', label='Predicted Years')
    plt.xticks(extended_years_range)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load the CSV file
df = pd.read_csv('liquid.csv')

# Specify the country and degree of the polynomial
country_name = 'Argentina'
degree = 2

# Define the range of years and extract the relevant data
years_range_adj = np.arange(2000, 2024, 2)
country_data = df[(df['Country Name'] == country_name) & 
                  (df['Indicator Name'] == 'CO2 emissions from liquid fuel consumption (kt)')]
y_data_range_adj = country_data[years_range_adj.astype(str)].values.flatten()

# Clean the data: Remove NaN values and get corresponding years
valid_indices_range_adj = ~np.isnan(y_data_range_adj)
x_data_range_clean_adj = years_range_adj[valid_indices_range_adj]
y_data_range_clean_adj = y_data_range_adj[valid_indices_range_adj]

# Fit the polynomial model to the cleaned data
poly_coefficients_range_clean_adj = fit_polynomial_model(x_data_range_clean_adj, y_data_range_clean_adj, degree)

# Calculate confidence intervals for the cleaned data
poly_confidence_intervals_range_clean_adj = err_ranges_polynomial(x_data_range_clean_adj, poly_coefficients_range_clean_adj, y_data_range_clean_adj, confidence=0.95)

# Extend the range of years for prediction (up to 2025) and predict future values
extended_years_range = np.arange(2000, 2026, 2)
extended_predicted_values = polynomial_model(extended_years_range, *poly_coefficients_range_clean_adj)

# Create an improved plot with predictions
create_plot(x_data_range_clean_adj, y_data_range_clean_adj, extended_predicted_values, poly_confidence_intervals_range_clean_adj, country_name, degree)



def polynomial_model(x, *coefficients):
    """
    Evaluate the polynomial model at given x values using the provided coefficients.

    Parameters:
    - x (array-like): Input values for the polynomial.
    - coefficients (float): Coefficients of the polynomial model.

    Returns:
    - array-like: Output values of the polynomial model.
    """
    return np.polyval(coefficients, x)

def fit_polynomial_model(x, y, degree):
    """
    Fit a polynomial model to the given data and return the coefficients.

    Parameters:
    - x (array-like): Independent variable values.
    - y (array-like): Dependent variable values.
    - degree (int): Degree of the polynomial model.

    Returns:
    - array-like: Coefficients of the fitted polynomial model.
    """
    coefficients = np.polyfit(x, y, degree)
    return coefficients

def err_ranges_polynomial(x, coefficients, y, confidence=0.95):
    """
    Calculate confidence intervals for a polynomial model.

    Parameters:
    - x (array-like): Independent variable values.
    - coefficients (array-like): Coefficients of the polynomial model.
    - y (array-like): Dependent variable values.
    - confidence (float): Confidence level for the interval.

    Returns:
    - tuple: Lower and upper bounds of the confidence interval.
    """
    p = np.poly1d(coefficients)
    y_fit = p(x)
    p_err = np.sqrt((1/(len(x)-len(coefficients))) * np.sum((y - y_fit)**2))
    delta = abs(t.ppf((1 - confidence) / 2., len(x)-len(coefficients))) * p_err
    upper = p(x) + delta
    lower = p(x) - delta
    return lower, upper

def create_plot(x_data, y_data, predicted_values, confidence_intervals, country_name, degree, indicator_name):
    """
    Create a plot with actual data, polynomial model, and confidence intervals.

    Parameters:
    - x_data (array-like): Independent variable values from the actual data.
    - y_data (array-like): Dependent variable values from the actual data.
    - predicted_values (array-like): Predicted values from the polynomial model.
    - confidence_intervals (tuple): Lower and upper bounds of the confidence interval.
    - country_name (str): Name of the country for plot title.
    - degree (int): Degree of the polynomial model.
    - indicator_name (str): Name of the CO2 emissions indicator.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 7))
    plt.scatter(x_data, y_data, color='blue', label='Actual Data')
    plt.plot(extended_years_range, predicted_values, label=f'Polynomial Model (Degree {degree})', color='green', linestyle='--')
    plt.fill_between(x_data, confidence_intervals[0], confidence_intervals[1], color='orange', alpha=0.2, label='95% Confidence Interval')

    for year in [2020, 2022]:
        plt.plot(year, polynomial_model(year, *poly_coefficients_range_clean_adj), 'ro')
        plt.text(year, polynomial_model(year, *poly_coefficients_range_clean_adj), f'  Predicted {year}', verticalalignment='bottom')

    plt.xlabel('Year')
    plt.ylabel(f'CO2 Emissions from {indicator_name} (kt)')
    plt.title(f'{indicator_name} in {country_name}\nPolynomial Model (Degree {degree}) and Predictions')
    plt.scatter([], [], color='red', marker='o', label='Predicted Years')
    plt.xticks(extended_years_range)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load the CSV file
df = pd.read_csv('gas.csv')

# Specify the country and degree of the polynomial
country_name = 'Argentina'
degree = 2

# Define the range of years and extract the relevant data
years_range_adj = np.arange(2000, 2024, 2)
indicator_name = 'CO2 emissions from gaseous fuel consumption (kt)'
country_data = df[(df['Country Name'] == country_name) ]
y_data_range_adj = country_data[years_range_adj.astype(str)].values.flatten()

# Clean the data: Remove NaN values and get corresponding years
valid_indices_range_adj = ~np.isnan(y_data_range_adj)
x_data_range_clean_adj = years_range_adj[valid_indices_range_adj]
y_data_range_clean_adj = y_data_range_adj[valid_indices_range_adj]

# Fit the polynomial model to the cleaned data
poly_coefficients_range_clean_adj = fit_polynomial_model(x_data_range_clean_adj, y_data_range_clean_adj, degree)

# Calculate confidence intervals for the cleaned data
poly_confidence_intervals_range_clean_adj = err_ranges_polynomial(x_data_range_clean_adj, poly_coefficients_range_clean_adj, y_data_range_clean_adj, confidence=0.95)

# Extend the range of years for prediction (up to 2025) and predict future values
extended_years_range = np.arange(2000, 2026, 2)
extended_predicted_values = polynomial_model(extended_years_range, *poly_coefficients_range_clean_adj)

# Create an improved plot with predictions
create_plot(x_data_range_clean_adj, y_data_range_clean_adj, extended_predicted_values, poly_confidence_intervals_range_clean_adj, country_name, degree, indicator_name)



def polynomial_model(x, *coefficients):
    """
    Evaluate the polynomial model at given x values using the provided coefficients.

    Parameters:
    - x (array-like): Input values for the polynomial.
    - coefficients (float): Coefficients of the polynomial model.

    Returns:
    - array-like: Output values of the polynomial model.
    """
    return np.polyval(coefficients, x)

def fit_polynomial_model(x, y, degree):
    """
    Fit a polynomial model to the given data and return the coefficients.

    Parameters:
    - x (array-like): Independent variable values.
    - y (array-like): Dependent variable values.
    - degree (int): Degree of the polynomial model.

    Returns:
    - array-like: Coefficients of the fitted polynomial model.
    """
    coefficients = np.polyfit(x, y, degree)
    return coefficients

def err_ranges_polynomial(x, coefficients, y, confidence=0.95):
    """
    Calculate confidence intervals for a polynomial model.

    Parameters:
    - x (array-like): Independent variable values.
    - coefficients (array-like): Coefficients of the polynomial model.
    - y (array-like): Dependent variable values.
    - confidence (float): Confidence level for the interval.

    Returns:
    - tuple: Lower and upper bounds of the confidence interval.
    """
    p = np.poly1d(coefficients)
    y_fit = p(x)
    p_err = np.sqrt((1/(len(x)-len(coefficients))) * np.sum((y - y_fit)**2))
    delta = abs(t.ppf((1 - confidence) / 2., len(x)-len(coefficients))) * p_err
    upper = p(x) + delta
    lower = p(x) - delta
    return lower, upper

def create_plot(x_data, y_data, predicted_values, confidence_intervals, country_name, degree, indicator_name):
    """
    Create a plot with actual data, polynomial model, and confidence intervals.

    Parameters:
    - x_data (array-like): Independent variable values from the actual data.
    - y_data (array-like): Dependent variable values from the actual data.
    - predicted_values (array-like): Predicted values from the polynomial model.
    - confidence_intervals (tuple): Lower and upper bounds of the confidence interval.
    - country_name (str): Name of the country for plot title.
    - degree (int): Degree of the polynomial model.
    - indicator_name (str): Name of the CO2 emissions indicator.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 7))
    plt.scatter(x_data, y_data, color='blue', label='Actual Data')
    plt.plot(extended_years_range, predicted_values, label=f'Polynomial Model (Degree {degree})', color='green', linestyle='--')
    plt.fill_between(x_data, confidence_intervals[0], confidence_intervals[1], color='red', alpha=0.2, label='95% Confidence Interval')

    for year in [2020, 2022]:
        plt.plot(year, polynomial_model(year, *poly_coefficients_range_clean_adj), 'ro')
        plt.text(year, polynomial_model(year, *poly_coefficients_range_clean_adj), f'  Predicted {year}', verticalalignment='bottom')

    plt.xlabel('Year')
    plt.ylabel(f'CO2 Emissions from {indicator_name} (kt)')
    plt.title(f'CO2 Emissions from {indicator_name} in {country_name}\nPolynomial Model (Degree {degree}) and Predictions')
    plt.scatter([], [], color='red', marker='o', label='Predicted Years')
    plt.xticks(extended_years_range)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load the CSV file
df = pd.read_csv('gas.csv')

# Specify the country and degree of the polynomial
country_name = 'Poland'
degree = 2

# Define the range of years and extract the relevant data
years_range_adj = np.arange(2000, 2024, 2)
indicator_name = 'CO2 emissions from gaseous fuel consumption (kt)'
country_data = df[(df['Country Name'] == country_name) & 
                  (df['Indicator Name'] == indicator_name)]
y_data_range_adj = country_data[years_range_adj.astype(str)].values.flatten()

# Clean the data: Remove NaN values and get corresponding years
valid_indices_range_adj = ~np.isnan(y_data_range_adj)
x_data_range_clean_adj = years_range_adj[valid_indices_range_adj]
y_data_range_clean_adj = y_data_range_adj[valid_indices_range_adj]

# Fit the polynomial model to the cleaned data
poly_coefficients_range_clean_adj = fit_polynomial_model(x_data_range_clean_adj, y_data_range_clean_adj, degree)

# Calculate confidence intervals for the cleaned data
poly_confidence_intervals_range_clean_adj = err_ranges_polynomial(x_data_range_clean_adj, poly_coefficients_range_clean_adj, y_data_range_clean_adj, confidence=0.95)

# Extend the range of years for prediction (up to 2025) and predict future values
extended_years_range = np.arange(2000, 2026, 2)
extended_predicted_values = polynomial_model(extended_years_range, *poly_coefficients_range_clean_adj)

# Create an improved plot with predictions
create_plot(x_data_range_clean_adj, y_data_range_clean_adj, extended_predicted_values, poly_confidence_intervals_range_clean_adj, country_name, degree, indicator_name)
