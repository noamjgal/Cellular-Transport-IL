import pandas as pd
import geopandas as gpd
from branca.colormap import LinearColormap
import pyproj
import folium
from folium.plugins import MarkerCluster

import os
import re

# Read in Shape of Zones
zones_path = '/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp'
zones = gpd.read_file(zones_path)

half_hour_path = '/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHalfHour6_20Trips201819_1270_weekday_v1.2.csv'
trip_data = pd.read_csv(half_hour_path)

# Create to_focus DataFrame
to_focus = trip_data[trip_data['ToZone'] == 101104].copy()

# Create from_focus DataFrame
from_focus = trip_data[trip_data['fromZone'] == 101104].copy()

# Add these paths at the beginning of your script
to_path = "/Users/noamgal/Downloads/To-From/ToFocus"
from_path = "/Users/noamgal/Downloads/To-From/FromFocus"

# Ensure the directories exist
os.makedirs(to_path, exist_ok=True)
os.makedirs(from_path, exist_ok=True)

def create_time_dict():
    time_dict = {}
    for hour in range(6, 20):  # 6 AM to 7:30 PM
        for minute in [0, 30]:
            key = f'h{hour}{minute:02d}'  # No leading zero for hour
            value = f'{hour:02d}:{minute:02d}'
            time_dict[key] = value
    return time_dict

# Create the time dictionary
time_dict = create_time_dict()

def create_interactive_map(zones, trip_data, focus_zone_id, direction='to', hour='h600'):
    # Determine the correct column names based on direction
    if direction == 'to':
        zone_column = 'fromZone'
    else:  # 'from'
        zone_column = 'ToZone'

    # Merge trip data with zones
    zones_with_data = zones.merge(trip_data, left_on='TAZ_1270', right_on=zone_column, how='left')
    zones_with_data[hour] = zones_with_data[hour].fillna(0)

    # Create a base map centered on the focus zone
    focus_zone = zones[zones['TAZ_1270'] == focus_zone_id]

    # Project to Israel TM Grid for accurate centroid calculation
    israel_tm_crs = pyproj.CRS.from_epsg(2039)  # Israel TM Grid
    focus_zone_tm = focus_zone.to_crs(israel_tm_crs)
    center_tm = focus_zone_tm.geometry.centroid.iloc[0]
    center = focus_zone_tm.to_crs(epsg=4326).geometry.centroid.iloc[0]

    m = folium.Map(location=[center.y, center.x], zoom_start=10)

    # Create a color map, excluding the focus zone from min/max calculation
    colors = ['#FEF0D9', '#FDD49E', '#FDBB84', '#FC8D59', '#EF6548', '#D7301F']
    vmin = zones_with_data[zones_with_data['TAZ_1270'] != focus_zone_id][hour].min()
    vmax = zones_with_data[zones_with_data['TAZ_1270'] != focus_zone_id][hour].max()
    colormap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    colormap.add_to(m)

    # Add zones to the map
    folium.GeoJson(
        zones_with_data,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties'][hour]) if feature['properties']['TAZ_1270'] != focus_zone_id else 'white',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['TAZ_1270', hour],
                                      aliases=['Zone ID', 'Trips'],
                                      localize=True)
    ).add_to(m)

    # Add labels for trips
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in zones_with_data.iterrows():
        if row['TAZ_1270'] != focus_zone_id:
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=f'<div style="font-size: 8pt;">{int(row[hour])}</div>'),
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

    # Calculate start and end times
    start_time = time_dict[hour]
    end_time = f"{int(start_time.split(':')[0]):02d}:{int(start_time.split(':')[1]) + 30:02d}"
    if end_time.endswith('60'):
        end_time = f"{int(end_time.split(':')[0]) + 1:02d}:00"

    # Add title with start and end times
    title_html = f'''
    <h3 align="center" style="font-size:16px">
        <b>Trips {'to' if direction == 'to' else 'from'} Focus Zone ({focus_zone_id}) from {start_time} to {end_time}</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Modify the save path based on direction
    if direction == 'to':
        save_path = os.path.join(to_path, f"trip_map_to_{hour}.html")
    else:  # 'from'
        save_path = os.path.join(from_path, f"trip_map_from_{hour}.html")

    # Save the map
    m.save(save_path)

# Assuming zones is already defined and in EPSG:4326 (WGS84) for Folium
zones = zones.to_crs(epsg=4326)

# Loop through all hours and create maps for both 'to' and 'from' directions
for hour in time_dict.keys():
    print(f"Generating map for {hour}...")
    create_interactive_map(zones, to_focus, 101104, direction='to', hour=hour)
    create_interactive_map(zones, from_focus, 101104, direction='from', hour=hour)

print("All maps generated!")



def extract_time(filename):
    match = re.search(r'h(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def create_merged_html(directory):
    # Get all HTML files, sort them by the time in their names
    html_files = sorted([f for f in os.listdir(directory) if f.endswith('.html')], 
                        key=extract_time)
    
    # Create the merged HTML content
    merged_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Merged Trip Maps</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            #controls { position: fixed; top: 10px; left: 10px; z-index: 1000; background: white; padding: 10px; }
            iframe { width: 100%; height: 100vh; border: none; }
        </style>
    </head>
    <body>
        <div id="controls">
            <button onclick="prevMap()">Previous</button>
            <span id="currentMap"></span>
            <button onclick="nextMap()">Next</button>
        </div>
        <script>
            const maps = [
    """
    
    for file in html_files:
        merged_content += f"        '{file}',\n"
    
    merged_content += """
            ];
            let currentIndex = 0;

            function updateMap() {
                document.getElementById('mapFrame').src = maps[currentIndex];
                document.getElementById('currentMap').textContent = `Map ${currentIndex + 1} of ${maps.length}`;
            }

            function nextMap() {
                currentIndex = (currentIndex + 1) % maps.length;
                updateMap();
            }

            function prevMap() {
                currentIndex = (currentIndex - 1 + maps.length) % maps.length;
                updateMap();
            }
        </script>
        <iframe id="mapFrame"></iframe>
        <script>updateMap();</script>
    </body>
    </html>
    """

    # Write the merged HTML to a file
    with open(os.path.join(directory, 'merged_maps.html'), 'w', encoding='utf-8') as f:
        f.write(merged_content)


create_merged_html(to_path)
print("Merged HTML file to Focus Zone created successfully.")
create_merged_html(from_path)
print("Merged HTML file from Focus Zone created successfully.")

