#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:41:30 2024

@author: noamgal
"""

import pandas as pd
import geopandas as gpd
import numpy as np

def inspect_data():
    print("Loading preprocessed data...")
    
    # Load the preprocessed data
    df_weekday = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/preprocessed_mobility_data.csv')
    zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/zones_3857.geojson')
    
    print("\nInspecting df_weekday:")
    print(f"Shape: {df_weekday.shape}")
    print(f"Columns: {df_weekday.columns.tolist()}")
    print("\nData types:")
    print(df_weekday.dtypes)
    
    print("\nChecking for missing values:")
    print(df_weekday.isnull().sum())
    
    print("\nSummary statistics for numerical columns:")
    print(df_weekday.describe())
    
    print("\nUnique values in 'fromZone' and 'ToZone':")
    print(f"fromZone: {df_weekday['fromZone'].nunique()}")
    print(f"ToZone: {df_weekday['ToZone'].nunique()}")
    
    print("\nChecking for negative values in 'distance' and 'trips_per_10k':")
    print(f"Negative distances: {(df_weekday['distance'] < 0).sum()}")
    print(f"Negative trips_per_10k: {(df_weekday['trips_per_10k'] < 0).sum()}")
    
    print("\nInspecting zones GeoDataFrame:")
    print(f"Shape: {zones.shape}")
    print(f"Columns: {zones.columns.tolist()}")
    print(f"CRS: {zones.crs}")
    
    print("\nChecking for missing geometries:")
    print(f"Missing geometries: {zones.geometry.isnull().sum()}")
    
    print("\nSample of df_weekday:")
    print(df_weekday.head())
    
    print("\nSample of zones:")
    print(zones.head())

if __name__ == "__main__":
    inspect_data()