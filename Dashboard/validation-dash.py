#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:58:46 2024

@author: noamgal
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def validate_data():
    print("Loading data...")
    preprocessed_data = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/preprocessed_mobility_data.csv')
    zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/zones_3857.geojson')
    
    print("\nData Validation Summary:")
    print(f"Total number of rows: {len(preprocessed_data)}")
    print(f"Number of unique focus zones: {preprocessed_data['focus_zone'].nunique()}")
    print(f"Number of unique TAZ_1270: {preprocessed_data['TAZ_1270'].nunique()}")
    
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

    # Compare with TAZ comparison approach
    print("\nComparing with TAZ comparison approach:")
    
    # Load original data (adjust paths as needed)
    population_df = pd.read_excel('/Users/noamgal/Downloads/NUR/celular1819_v1.3/1270_population.xlsx')
    df_weekday = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_v1.csv')
    df_weekday_arrival = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_arrival_v1.2.csv')

    # Check population data
    print("\nPopulation data check:")
    print(f"Number of zones in population data: {len(population_df)}")
    print(f"Number of zones in preprocessed data: {preprocessed_data['TAZ_1270'].nunique()}")
    
    # Check total trips
    print("\nTotal trips check:")
    focus_zone = preprocessed_data['focus_zone'].min()  # Use the first focus zone as an example
    original_trips = df_weekday[df_weekday['ToZone'] == focus_zone].groupby('fromZone')[['h' + str(i) for i in range(24)]].sum().sum(axis=1)
    preprocessed_trips = preprocessed_data[preprocessed_data['focus_zone'] == focus_zone]['total_trips']
    
    print(f"Total trips for focus zone {focus_zone}:")
    print(f"  Original data: {original_trips.sum()}")
    print(f"  Preprocessed data: {preprocessed_trips.sum()}")
    
    # Check time signature
    print("\nTime signature check:")
    original_arrivals = df_weekday_arrival[df_weekday_arrival['ToZone'] == focus_zone].iloc[0]
    original_departures = df_weekday[df_weekday['fromZone'] == focus_zone].iloc[0]
    preprocessed_zone_data = preprocessed_data[preprocessed_data['focus_zone'] == focus_zone].iloc[0]
    
    for i in range(24):
        original_arrival = original_arrivals[f'h{i}']
        original_departure = original_departures[f'h{i}']
        preprocessed_arrival = preprocessed_zone_data[f'arrivals_h{i}']
        preprocessed_departure = preprocessed_zone_data[f'departures_h{i}']
        
        print(f"Hour {i}:")
        print(f"  Arrivals - Original: {original_arrival}, Preprocessed: {preprocessed_arrival}")
        print(f"  Departures - Original: {original_departure}, Preprocessed: {preprocessed_departure}")
    
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