import pandas as pd
import geopandas as gpd
import numpy as np

# Load data (adjust paths as needed)
print("Loading data...")
zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp').to_crs(epsg=3857)
population_df = pd.read_excel('/Users/noamgal/Downloads/NUR/celular1819_v1.3/1270_population.xlsx')
df_weekday = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_v1.csv')
df_weekday_arrival = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_arrival_v1.2.csv')

print(f"Zones shape: {zones.shape}")
print(f"Population data shape: {population_df.shape}")
print(f"Weekday trips shape: {df_weekday.shape}")
print(f"Weekday arrivals shape: {df_weekday_arrival.shape}")

# Prepare population data
print("\nPreparing population data...")
population = population_df.set_index('TAZ_1270')[2019]
mean_population = population[population > 0].mean()
population = population.replace(0, mean_population)
print(f"Mean population: {mean_population:.2f}")

# Create preprocessed data
print("\nCreating preprocessed data...")
preprocessed_data = []

for focus_zone in zones['TAZ_1270']:
    # Calculate trips to focus zone
    trips_to_focus = df_weekday[df_weekday['ToZone'] == focus_zone].groupby('fromZone')[['h' + str(i) for i in range(24)]].sum().sum(axis=1)
    
    # Prepare data for this focus zone
    zone_data = pd.DataFrame({
        'TAZ_1270': zones['TAZ_1270'],
        'focus_zone': focus_zone,
        'population': population,
        'total_trips': trips_to_focus
    })
    
    # Calculate distances
    focus_point = zones[zones['TAZ_1270'] == focus_zone].geometry.centroid.iloc[0]
    zone_data['distance'] = zones.geometry.centroid.distance(focus_point) / 1000  # Convert to km
    
    # Calculate trips per 10k population
    zone_data['trips_per_10k'] = (zone_data['total_trips'] / zone_data['population']) * 10000
    
    # Add hourly data
    for i in range(24):
        zone_data[f'arrivals_h{i}'] = df_weekday_arrival[df_weekday_arrival['ToZone'] == focus_zone][f'h{i}'].values[0]
        zone_data[f'departures_h{i}'] = df_weekday[df_weekday['fromZone'] == focus_zone][f'h{i}'].sum()
    
    preprocessed_data.append(zone_data)
    
    if len(preprocessed_data) % 100 == 0:
        print(f"Processed {len(preprocessed_data)} zones")

# Combine all preprocessed data
all_preprocessed_data = pd.concat(preprocessed_data, ignore_index=True)
all_preprocessed_data = all_preprocessed_data.fillna(0)  # Replace NaN with 0

print(f"\nAll preprocessed data shape: {all_preprocessed_data.shape}")
print(f"Columns in preprocessed data: {all_preprocessed_data.columns.tolist()}")
print(f"\nSample of final preprocessed data:\n{all_preprocessed_data.head()}")

# Calculate and print summary statistics
print("\nSummary Statistics:")
print(f"Average distance: {all_preprocessed_data['distance'].mean():.2f} km")
print(f"Average total trips: {all_preprocessed_data['total_trips'].mean():.2f}")
print(f"Average trips per 10k: {all_preprocessed_data['trips_per_10k'].mean():.2f}")
print(f"Max trips per 10k: {all_preprocessed_data['trips_per_10k'].max():.2f}")

# Save preprocessed data
print("\nSaving preprocessed data...")
all_preprocessed_data.to_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/preprocessed_mobility_data.csv', index=False)

# Save zones geodataframe for later use
print("Saving zones data...")
zones.to_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/zones_3857.geojson', driver='GeoJSON')

print("\nPreprocessing complete. Data saved to preprocessed_mobility_data.csv and zones_3857.geojson")