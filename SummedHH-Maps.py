#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:51:38 2024

@author: noamgal
"""
import pandas as pd
import geopandas as gpd
from branca.colormap import LinearColormap
import pyproj
import folium
from folium.plugins import MarkerCluster



# Loads and reads the Half Hour Trip Data
half_hour_path = '/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHalfHour6_20Trips201819_1270_weekday_v1.2.csv'
travel_hh = pd.read_csv(half_hour_path)

# Shapefile Paths
zones_path = '/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp'
# Read the shapefile
zones = gpd.read_file(zones_path)

# Create to_focus DataFrame
to_focus = travel_hh[travel_hh['ToZone'] == 101104].copy()

# Create from_focus DataFrame
from_focus = travel_hh[travel_hh['fromZone'] == 101104].copy()

# Sum across all hour columns for to_focus
hour_columns = [col for col in to_focus.columns if col.startswith('h')]
to_focus_totals = to_focus.groupby('fromZone')[hour_columns].sum().sum(axis=1).reset_index()
to_focus_totals.columns = ['fromZone', 'TotalTrips']
to_focus_totals = to_focus_totals.sort_values('TotalTrips', ascending=False)

# Sum across all hour columns for from_focus
from_focus_totals = from_focus.groupby('ToZone')[hour_columns].sum().sum(axis=1).reset_index()
from_focus_totals.columns = ['ToZone', 'TotalTrips']
from_focus_totals = from_focus_totals.sort_values('TotalTrips', ascending=False)

def create_interactive_map(zones, trip_data, focus_zone_id, direction='to'):
    # Determine the correct column names based on direction
    if direction == 'to':
        zone_column = 'fromZone'
    else:  # 'from'
        zone_column = 'ToZone'

    # Merge trip data with zones
    zones_with_data = zones.merge(trip_data, left_on='TAZ_1270', right_on=zone_column, how='left')
    zones_with_data['TotalTrips'] = zones_with_data['TotalTrips'].fillna(0)

    # Create a base map centered on the focus zone
    focus_zone = zones[zones['TAZ_1270'] == focus_zone_id]

    # Project to Israel TM Grid for accurate centroid calculation
    israel_tm_crs = pyproj.CRS.from_epsg(2039)  # Israel TM Grid
    center = focus_zone.to_crs(epsg=4326).geometry.centroid.iloc[0]

    m = folium.Map(location=[center.y, center.x], zoom_start=10)

    # Create a color map, excluding the focus zone from min/max calculation
    colors = ['#FEF0D9', '#FDD49E', '#FDBB84', '#FC8D59', '#EF6548', '#D7301F']
    vmin = zones_with_data[zones_with_data['TAZ_1270'] != focus_zone_id]['TotalTrips'].min()
    vmax = zones_with_data[zones_with_data['TAZ_1270'] != focus_zone_id]['TotalTrips'].max()
    colormap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    colormap.add_to(m)

    # Add zones to the map
    folium.GeoJson(
        zones_with_data,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['TotalTrips']) if feature['properties']['TAZ_1270'] != focus_zone_id else 'white',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['TAZ_1270', 'TotalTrips'],
                                      aliases=['Zone ID', 'Total Trips'],
                                      localize=True)
    ).add_to(m)

    # Add labels for total trips
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in zones_with_data.iterrows():
        if row['TAZ_1270'] != focus_zone_id:
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=f'<div style="font-size: 8pt;">{int(row.TotalTrips)}</div>'),
            ).add_to(marker_cluster)

    # Highlight focus zone
    focus_zone_style = {
        'color': 'red',
        'weight': 3,
        'fillOpacity': 0,
    }
    folium.GeoJson(focus_zone, style_function=lambda x: focus_zone_style).add_to(m)

    # Add a star marker for the focus zone
    folium.Marker(
        location=[center.y, center.x],
        icon=folium.Icon(color='orange', icon='star'),
    ).add_to(m)

    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px">
        <b>Trips {'to' if direction == 'to' else 'from'} Focus Zone ({focus_zone_id})</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save the map
    m.save(f"/Users/noamgal/Downloads/Test2interactive_trip_map_{direction}.html")

# Assuming zones is already defined and in EPSG:4326 (WGS84) for Folium
zones = zones.to_crs(epsg=4326)

# Create the interactive map for trips to the focus zone
create_interactive_map(zones, to_focus_totals, 101104, direction='to')

# Create the interactive map for trips from the focus zone
create_interactive_map(zones, from_focus_totals, 101104, direction='from')