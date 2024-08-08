#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:55:05 2024

@author: noamgal
"""

import pandas as pd
import geopandas as gpd
import shapely.geometry
import requests
from datetime import datetime
import time
import os
import random

# To use this script, you must have built an OTP model stored locally with Israel's GTFS and OSM data
# OTP Documentation: https://docs.opentripplanner.org/en/latest/
# OTP Version Releases: https://github.com/opentripplanner/OpenTripPlanner/releases
# GTFS Data can be downloaded here: https://gtfs.mot.gov.il/gtfsfiles/
# OSM data can be downloaded here: https://download.geofabrik.de/asia/israel-and-palestine.html

# the output of this 


# Load TAZ zones shapefile
zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp')
zones = zones.to_crs(epsg=4326)

# OTP GTFS GraphQL local API endpoint
OTP_URL = "http://localhost:8080/otp/routers/default/index/graphql"

def get_travel_time(from_lat, from_lon, to_lat, to_lon, mode, departure_time):
    query = """
    query ($from: InputCoordinates!, $to: InputCoordinates!, $date: String!, $time: String!, $mode: Mode!) {
      plan(
        from: $from
        to: $to
        date: $date
        time: $time
        transportModes: [{mode: $mode}]
      ) {
        itineraries {
          duration
        }
      }
    }
    """
    
    variables = {
        "from": {"lat": from_lat, "lon": from_lon},
        "to": {"lat": to_lat, "lon": to_lon},
        "date": departure_time.strftime("%Y-%m-%d"),
        "time": departure_time.strftime("%H:%M:%S"),
        "mode": "CAR" if mode == "AUTO" else "TRANSIT"
    }
    
    headers = {
        'Content-Type': 'application/json',
        'OTPTimeout': '180000'
    }
    
    try:
        response = requests.post(OTP_URL, json={"query": query, "variables": variables}, headers=headers)
        
        if response.status_code != 200:
            print(f"Error response (status {response.status_code}): {response.text}")
            return None

        data = response.json()
        if 'data' in data and 'plan' in data['data'] and data['data']['plan']['itineraries']:
            return data['data']['plan']['itineraries'][0]['duration'] / 60  # Convert seconds to minutes
        else:
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
    
    return None



def generate_valid_point(geometry, max_attempts=50):
    for _ in range(max_attempts):
        minx, miny, maxx, maxy = geometry.bounds
        point = shapely.geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if geometry.contains(point):
            return point
    return None

def calculate_travel_times(focus_zone, zones, mode, direction, output_dir):
    time_matrix = pd.DataFrame(columns=['TAZ_1270', 'TravelTime'])
    total = len(zones)
    count = 0
    valid_count = 0
    # For trips to the focus zone, we analyze departure time at 7:30
    # For trip from the focus zone, we analyze departure time at 17:00
    if direction == 'to':
        departure_time = datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
    else:
        departure_time = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
    
    focus_zone_row = zones[zones['TAZ_1270'] == focus_zone]
    if focus_zone_row.empty:
        raise KeyError(f"Focus zone {focus_zone} not found in TAZ_1270 column.")
    
    focus_lat, focus_lon = focus_zone_row.iloc[0].geometry.centroid.y, focus_zone_row.iloc[0].geometry.centroid.x
    
    for _, zone in zones.iterrows():
        travel_time = None
        attempts = 0
        zone_points = [zone.geometry.centroid]
        
        while travel_time is None and attempts < 10:
            if attempts >= len(zone_points):
                new_point = generate_valid_point(zone.geometry)
                if new_point:
                    zone_points.append(new_point)
                else:
                    print(f"Failed to generate new point for zone {zone['TAZ_1270']} after multiple attempts")
                    break
            
            current_point = zone_points[attempts]
            if direction == 'to':
                from_lat, from_lon = current_point.y, current_point.x
                to_lat, to_lon = focus_lat, focus_lon
            else:
                from_lat, from_lon = focus_lat, focus_lon
                to_lat, to_lon = current_point.y, current_point.x
            
            travel_time = get_travel_time(from_lat, from_lon, to_lat, to_lon, mode, departure_time)
            attempts += 1
        
        if travel_time is not None:
            new_row = pd.DataFrame({'TAZ_1270': [zone['TAZ_1270']], 'TravelTime': [travel_time]})
            time_matrix = pd.concat([time_matrix, new_row], ignore_index=True, axis=0)
            valid_count += 1
        else:
            print(f"No valid travel time found for zone {zone['TAZ_1270']} after {attempts} attempts.")
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} destinations, {valid_count} valid times")
        time.sleep(0.1)  # To avoid overwhelming the API
    
    print(f"Total valid travel times: {valid_count}/{total}")
    
    # Save the results
    filename = f"{focus_zone}_{direction}_{mode}_travel_times.csv"
    filepath = os.path.join(output_dir, filename)
    time_matrix.to_csv(filepath, index=False)
    print(f"Travel times saved to {filepath}")
    
    return time_matrix


# Ask user for input
# Beer Sheva Innovation District focus zone input is 101104
while True:
    user_input = input("Please enter the focus zone ID: ")
    try:
        focus_zone = int(user_input)
        if focus_zone in zones['TAZ_1270'].values:
            break
        else:
            print(f"Zone {focus_zone} not found in the TAZ_1270 column. Please try again.")
    except ValueError:
        print("Please enter a valid integer for the zone ID.")

print(f"Using focus zone: {focus_zone}")

# The rest of your code follows...
output_dir = "/Users/noamgal/Downloads/NUR/celular1819_v1.3"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Calculating travel times to focus zone {focus_zone}...")
calculate_travel_times(focus_zone, zones, "AUTO", "to", output_dir)
calculate_travel_times(focus_zone, zones, "TRANSIT", "to", output_dir)

print(f"Calculating travel times from focus zone {focus_zone}...")
calculate_travel_times(focus_zone, zones, "AUTO", "from", output_dir)
calculate_travel_times(focus_zone, zones, "TRANSIT", "from", output_dir)

print("All calculations complete.")
