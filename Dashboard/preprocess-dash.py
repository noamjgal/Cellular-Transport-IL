import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm

print("Loading data...")
zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp').to_crs(epsg=3857)
population_df = pd.read_excel('/Users/noamgal/Downloads/NUR/celular1819_v1.3/1270_population.xlsx')
df_weekday = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_v1.csv')
df_weekday_arrival = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_arrival_v1.2.csv')

print("Preprocessing data...")
print("Step 1/5: Preparing population data")
population = population_df.set_index('TAZ_1270')[2019]
mean_population = population[population > 0].mean()
population = population.replace(0, mean_population)

print("Step 2/5: Adding population to df_weekday")
df_weekday['population'] = df_weekday['fromZone'].map(population)

print("Step 3/5: Calculating total trips")
hour_cols = [f'h{i}' for i in range(24)]
df_weekday['total_trips'] = df_weekday[hour_cols].sum(axis=1)

print("Step 4/5: Calculating trips per 10k population")
df_weekday['trips_per_10k'] = (df_weekday['total_trips'] / df_weekday['population']) * 10000
df_weekday['trips_per_10k'] = df_weekday['trips_per_10k'].replace([np.inf, -np.inf], np.nan).fillna(0)

print("Step 5/5: Calculating distances (this may take a while)")
zone_centroids = zones.set_index('TAZ_1270').geometry.centroid

def calculate_distance(row):
    try:
        from_centroid = zone_centroids.loc[row['fromZone']]
        to_centroid = zone_centroids.loc[row['ToZone']]
        return from_centroid.distance(to_centroid) / 1000
    except KeyError:
        print(f"KeyError: fromZone {row['fromZone']} or ToZone {row['ToZone']} not found in centroids")
        return np.nan

tqdm.pandas(desc="Calculating distances")
df_weekday['distance'] = df_weekday.progress_apply(calculate_distance, axis=1)

print(f"\nPreprocessed data shape: {df_weekday.shape}")
print(f"Columns in preprocessed data: {df_weekday.columns.tolist()}")
print(f"\nSample of preprocessed data:\n{df_weekday.head()}")

print("\nSummary Statistics:")
print(f"Average distance: {df_weekday['distance'].mean():.2f} km")
print(f"Average total trips: {df_weekday['total_trips'].mean():.2f}")
print(f"Average trips per 10k: {df_weekday['trips_per_10k'].mean():.2f}")
print(f"Max trips per 10k: {df_weekday['trips_per_10k'].max():.2f}")

print("\nSaving preprocessed data...")
df_weekday.to_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/preprocessed_mobility_data.csv', index=False)

print("Saving zones data...")
zones.to_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/zones_3857.geojson', driver='GeoJSON')

print("\nPreprocessing complete. Data saved to preprocessed_mobility_data.csv and zones_3857.geojson")