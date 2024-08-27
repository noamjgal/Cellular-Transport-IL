import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def validate_data():
    print("Loading data...")
    preprocessed_data = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/preprocessed_mobility_data.csv')
    
    print("\nData Validation Summary:")
    print(f"Total number of rows: {len(preprocessed_data)}")
    print(f"Number of unique fromZones: {preprocessed_data['fromZone'].nunique()}")
    print(f"Number of unique ToZones: {preprocessed_data['ToZone'].nunique()}")
    
    print("\nColumn statistics:")
    for column in ['population', 'total_trips', 'distance', 'trips_per_10k']:
        print(f"\n{column}:")
        print(f"  Min: {preprocessed_data[column].min()}")
        print(f"  Max: {preprocessed_data[column].max()}")
        print(f"  Mean: {preprocessed_data[column].mean()}")
        print(f"  Median: {preprocessed_data[column].median()}")
        print(f"  Number of zeros: {(preprocessed_data[column] == 0).sum()}")
        print(f"  Number of NaN: {preprocessed_data[column].isna().sum()}")
    
    print("\nChecking for invalid calculations:")
    invalid_trips_per_10k = ((preprocessed_data['trips_per_10k'] == np.inf) | (preprocessed_data['trips_per_10k'].isna())).sum()
    print(f"Number of invalid trips_per_10k: {invalid_trips_per_10k}")

    if invalid_trips_per_10k > 0:
        print("Sample of rows with invalid trips_per_10k:")
        print(preprocessed_data[(preprocessed_data['trips_per_10k'] == np.inf) | (preprocessed_data['trips_per_10k'].isna())].head())

    print("\nSample of preprocessed data:")
    print(preprocessed_data.head())

    # Visualize trips distribution
    plt.figure(figsize=(12, 6))
    plt.hist(preprocessed_data['total_trips'], bins=50, edgecolor='black')
    plt.title('Distribution of Total Trips')
    plt.xlabel('Number of Trips')
    plt.ylabel('Frequency')
    plt.savefig('/Users/noamgal/Downloads/NUR/celular1819_v1.3/trips_distribution.png')
    plt.close()
    
    print("\nTrips distribution plot saved as 'trips_distribution.png'")

if __name__ == "__main__":
    validate_data()