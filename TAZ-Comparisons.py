
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:43:48 2024

@author: noamgal
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar

print('imports completed')

# Load data (adjust paths as needed)
zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp').to_crs(epsg=3857)
large_zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/33_02.09.2021.shp').to_crs(epsg=3857)
population_df = pd.read_excel('/Users/noamgal/Downloads/NUR/celular1819_v1.3/1270_population.xlsx')
df_weekday = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_v1.csv')
df_weekday_arrival = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_arrival_v1.2.csv')

print('datasets loaded')

def generate_maps_and_graphs(focus_zone):
    print(f"Generating maps and graphs for focus zone: {focus_zone}")
    
    # Prepare data
    population = population_df.set_index('TAZ_1270')[2019]
    mean_population = population[population > 0].mean()
    population = population.replace(0, mean_population)
    
    trips_to_focus = df_weekday[df_weekday['ToZone'] == focus_zone].groupby('fromZone')[['h' + str(i) for i in range(24)]].sum().sum(axis=1)
    
    # Merge population and trip data with zones
    zones_data = zones.merge(
        pd.DataFrame({
            'TAZ_1270': population.index, 
            'population': population,
            'total_trips': trips_to_focus
        }), 
        on='TAZ_1270', 
        how='left'
    )
    zones_data['total_trips'] = zones_data['total_trips'].fillna(0)
    zones_data['population'] = zones_data['population'].fillna(mean_population)  # Fill NaN population with mean
    zones_data['trips_per_10k'] = (zones_data['total_trips'] / zones_data['population']) * 10000
    
    # Replace inf and NaN values with 0
    zones_data['trips_per_10k'] = zones_data['trips_per_10k'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Identify the large zone (TAZ_33) containing the focus zone
    focus_large_zone = zones_data[zones_data['TAZ_1270'] == focus_zone]['TAZ_33'].iloc[0]
    
    # Identify neighboring large zones
    neighboring_large_zones = large_zones[large_zones.touches(large_zones[large_zones['TAZ_33'] == focus_large_zone].geometry.iloc[0])]['TAZ_33'].tolist()
    neighboring_large_zones.append(focus_large_zone)
    
    # Separate zones into detailed (1270) and aggregated (33)
    detailed_zones = zones_data[zones_data['TAZ_33'].isin(neighboring_large_zones)]
    aggregated_zones = zones_data[~zones_data['TAZ_33'].isin(neighboring_large_zones)]
    
    # Aggregate data for larger zones
    aggregated_data = aggregated_zones.dissolve(by='TAZ_33', aggfunc={
        'population': 'sum',
        'total_trips': 'sum',
        'AREA': 'sum'
    }).reset_index()
    aggregated_data['trips_per_10k'] = (aggregated_data['total_trips'] / aggregated_data['population']) * 10000
    
    # Replace inf and NaN values with 0
    aggregated_data['trips_per_10k'] = aggregated_data['trips_per_10k'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Combine detailed and aggregated zones
    combined_zones = pd.concat([detailed_zones, aggregated_data])
    
    print(f"Number of zones after processing: {len(combined_zones)}")
    print(f"Number of zones with 0 trips_per_10k: {(combined_zones['trips_per_10k'] == 0).sum()}")
    print(f"Range of trips_per_10k: {combined_zones['trips_per_10k'].min()} to {combined_zones['trips_per_10k'].max()}")
   

    # Create Folium map
    focus_zone_geom = zones[zones['TAZ_1270'] == focus_zone].to_crs(epsg=3857)
    focus_zone_center = focus_zone_geom.geometry.centroid.iloc[0]
    focus_zone_center_4326 = focus_zone_geom.to_crs(epsg=4326).geometry.centroid.iloc[0]
    m = folium.Map(location=[focus_zone_center_4326.y, focus_zone_center_4326.x], zoom_start=9)


    # Set up color map
    vmin = combined_zones['trips_per_10k'].min()
    vmax = np.percentile(combined_zones['trips_per_10k'], 95)
    print(f"Colormap range: {vmin} to {vmax}")
    colormap = LinearColormap(colors=['#FFFFD4', '#FEE391', '#D9F0A3', '#ADDD8E', '#78C679', '#31A354'], vmin=vmin, vmax=vmax)
    
    def style_function(feature):
        trips = feature['properties']['trips_per_10k']
        if trips is None or trips == 0:
            return {'fillColor': 'gray', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
        elif trips > vmax:
            return {'fillColor': '#006400', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
        return {'fillColor': colormap(trips), 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
    

    # Add a function to prepare tooltip content
    def prepare_tooltip(row):
        if pd.notnull(row['TAZ_1270']):
            zone_info = f"TAZ_1270: {int(row['TAZ_1270'])}"
        else:
            zone_info = f"TAZ_33: {int(row['TAZ_33'])}"
        
        return f"""
        <table>
            <tr><th>{zone_info}</th></tr>
            <tr><th>Population:</th><td>{int(row['population'])}</td></tr>
            <tr><th>Total Trips:</th><td>{int(row['total_trips'])}</td></tr>
            <tr><th>Trips per 10k:</th><td>{row['trips_per_10k']:.2f}</td></tr>
        </table>
        """

    # Add tooltip content to combined_zones
    combined_zones['tooltip_content'] = combined_zones.apply(prepare_tooltip, axis=1)

    # Add GeoJson layers
    folium.GeoJson(
        combined_zones.to_crs(epsg=4326),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['tooltip_content'],
            aliases=[''],
            labels=False,
            sticky=False,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    
    # Add markers with labels
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in combined_zones.iterrows():
        if row['TAZ_1270'] != focus_zone:
            centroid = row.geometry.centroid
            trips_value = row['trips_per_10k']
            population = int(row['population']) if not np.isnan(row['population']) else 0
            total_trips = int(row['total_trips']) if not np.isnan(row['total_trips']) else 0
            
            # Determine if this is a close zone or far zone
            is_close_zone = 'TAZ_1270' in row.index and not pd.isna(row['TAZ_1270'])
            
            if is_close_zone:
                zone_id = f"TAZ_1270: {int(row['TAZ_1270'])}"
            else:
                zone_id = f"TAZ_33: {int(row['TAZ_33'])}"
            
            label = f'{zone_id}<br>Pop: {population}<br>Trips: {total_trips}<br>Per 10k: {int(trips_value)}'
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=f'<div style="font-size: 8pt; white-space: nowrap;">{label}</div>'),
            ).add_to(marker_cluster)
    
    # Highlight focus zone
    focus_zone_style = {
        'color': 'red',
        'weight': 3,
        'fillOpacity': 0,
    }
    folium.GeoJson(zones_data[zones_data['TAZ_1270'] == focus_zone].to_crs(epsg=4326), style_function=lambda x: focus_zone_style).add_to(m)
    

    # Add a star marker for the focus zone
    folium.Marker(
        location=[focus_zone_center_4326.y, focus_zone_center_4326.x],
        icon=folium.Icon(color='orange', icon='star'),
    ).add_to(m)

    # Add the colormap to the map
    colormap.add_to(m)

    # Add a custom legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                width: 220px; height: 160px; 
                background-color: white; 
                border-radius: 5px; 
                padding: 10px; 
                z-index: 9999;">
        <p style="font-size:14px; margin-bottom: 5px;"><b>Legend</b></p>
        <p style="font-size:12px; margin-top: 0;">Trips per 10,000 people</p>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="background: linear-gradient(to right, #FFFFD4, #FEE391, #D9F0A3, #ADDD8E, #78C679, #31A354); 
                        width: 150px; height: 20px;"></div>
            <div style="display: flex; justify-content: space-between; width: 150px;">
                <span style="font-size: 10px;">{:.0f}</span>
                <span style="font-size: 10px;">{:.0f}</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="background-color: #006400; width: 20px; height: 20px;"></div>
            <span style="font-size: 10px; margin-left: 5px;">> {:.0f} (95th percentile)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="background-color: gray; width: 20px; height: 20px;"></div>
            <span style="font-size: 10px; margin-left: 5px;">No recorded trips</span>
        </div>
    </div>
    '''.format(vmin, vmax, vmax)

    m.get_root().html.add_child(folium.Element(legend_html))

    # Add title and legend explanation
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 800px; 
                background-color: white; 
                border-radius: 5px; 
                padding: 10px; 
                z-index: 9999;">
        <h3 align="center" style="font-size:16px; margin-bottom: 5px;">
            <b>Trips to Focus Zone ({focus_zone})</b>
        </h3>
        <p style="font-size:12px; margin-top: 5px;">
            Legend shows trips per 10,000 people. Colors range from light yellow (fewer trips) 
            to dark green (more trips), capped at the 95th percentile. 
            Zones with trips above the 95th percentile are colored in a darker green. 
            Gray zones have no recorded trips to the focus zone.
            Detailed zones (TAZ_1270) are shown for the focus zone and its immediate neighbors, 
            while aggregated zones (TAZ_33) are used for areas further away.
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save Folium map
    output_path = f'/Users/noamgal/Downloads/NUR/celular1819_v1.3/focus_zone_{focus_zone}_map.html'
    m.save(output_path)
    print(f"Map saved to: {output_path}")
    
    # Generate graphs
    print("Calculating distances...")
    trips_with_distance = calculate_distances(combined_zones, focus_zone, trips_to_focus)
    print(f"Shape of trips_with_distance: {trips_with_distance.shape}")
    print(f"Columns in trips_with_distance: {trips_with_distance.columns}")
    print(f"Sample of trips_with_distance:\n{trips_with_distance.head()}")
    print(f"Sum of TotalTrips: {trips_with_distance['TotalTrips'].sum()}")
    
    estimate_district_population(df_weekday, df_weekday_arrival, focus_zone)
    plot_trips_by_distance(trips_with_distance, focus_zone)
    plot_time_signature(df_weekday, df_weekday_arrival, focus_zone)
    plot_focus_zone_map(focus_zone)

    print(f"Maps and graphs for focus zone {focus_zone} have been generated.")
    


def calculate_distances(zones_data, focus_zone, trips):
    print(f"Calculating distances for {len(zones_data)} zones")
    
    focus_point = zones_data[zones_data['TAZ_1270'] == focus_zone].geometry.centroid.iloc[0]
    
    # Ensure we're using TAZ-1270 for all zones
    all_zones = zones.copy()
    all_zones['distance'] = all_zones.geometry.centroid.distance(focus_point) / 1000  # Convert to km
    
    # Merge with trips data
    result = all_zones[['TAZ_1270', 'distance']].merge(trips.reset_index(), left_on='TAZ_1270', right_on='fromZone', how='right')
    result['TotalTrips'] = result[0]  # Rename the trips column
    result = result.drop(columns=['fromZone', 0])  # Drop unnecessary columns
    
    # Remove the focus zone itself
    result = result[result['TAZ_1270'] != focus_zone]
    
    print(f"Shape of result after removing focus zone: {result.shape}")
    print(f"Sample of result:\n{result.head()}")
    print(f"Distance range: {result['distance'].min()} to {result['distance'].max()} km")
    print(f"Sum of TotalTrips: {result['TotalTrips'].sum()}")
    
    return result

def plot_trips_by_distance(trips_with_distance, focus_zone):
    print("Plotting trips by distance...")
    
    max_distance = 150
    filtered_trips = trips_with_distance[trips_with_distance['distance'] <= max_distance].dropna()
    
    print(f"Shape of filtered_trips: {filtered_trips.shape}")
    print(f"Sum of TotalTrips in filtered_trips: {filtered_trips['TotalTrips'].sum()}")
    
    # Create histogram with 5 km increments
    bins = np.arange(0, max_distance + 5, 5)
    
    def create_histogram(data, weights, bins, title_suffix, y_label, filename_suffix):
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()  # Create a second y-axis
        sns.set_style("whitegrid")
        
        n, bins, patches = ax1.hist(data, weights=weights, bins=bins, edgecolor='black', color='skyblue')
        
        # Calculate cumulative percentage
        cumulative = np.cumsum(n)
        cumulative_percent = cumulative / cumulative[-1] * 100
        
        # Plot cumulative percentage line
        ax2.plot(bins[1:], cumulative_percent, color='darkblue', linewidth=2)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Cumulative Percentage', fontsize=24, fontweight='bold')
        
        ax1.set_title(f'Histogram of Trips to Zone {focus_zone} by Distance {title_suffix}\n(excluding intra-zonal trips)', fontsize=28, fontweight='bold')
        ax1.set_xlabel('Distance (km)', fontsize=24, fontweight='bold')
        ax1.set_ylabel(y_label, fontsize=24, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='y', which='major', labelsize=12)
        ax1.set_xlim(0, max_distance)
        
        if len(filtered_trips) > 0:
            mean_distance = np.average(filtered_trips['distance'], weights=filtered_trips['TotalTrips'])
            median_distance = np.median(np.repeat(filtered_trips['distance'], filtered_trips['TotalTrips'].astype(int)))
            total_trips = filtered_trips['TotalTrips'].sum()
            
            stats_text = f'Mean Trip Distance: {mean_distance:.2f} km\n' \
                         f'Median Trip Distance: {median_distance:.2f} km\n' \
                         f'Total Trips: {total_trips:,.0f}'
            ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=18,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        output_path = f'/Users/noamgal/Downloads/NUR/celular1819_v1.3/trips_by_distance_histogram_{filename_suffix}_zone_{focus_zone}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trips by distance histogram ({filename_suffix}) saved to: {output_path}")

    # Create histogram for total trips
    create_histogram(filtered_trips['distance'], 
                     filtered_trips['TotalTrips'], 
                     bins,
                     '(Total Trips)', 
                     'Number of Trips', 
                     'total')

    # Create histogram for percentage of trips
    total_trips = filtered_trips['TotalTrips'].sum()
    percentage_weights = filtered_trips['TotalTrips'] / total_trips * 100
    create_histogram(filtered_trips['distance'], 
                     percentage_weights, 
                     bins,
                     '(Percentage of Trips)', 
                     'Percentage of Trips', 
                     'percentage')

def plot_time_signature(df_weekday, df_weekday_arrival, focus_zone):
    to_focus = df_weekday_arrival[df_weekday_arrival['ToZone'] == focus_zone]
    from_focus = df_weekday[df_weekday['fromZone'] == focus_zone]
    
    time_columns = [col for col in to_focus.columns if col.startswith('h')]
    arrivals = to_focus[time_columns].sum()
    departures = from_focus[time_columns].sum()
    
    hours = range(24)
    x = np.arange(24)
    width = 0.35
    
    def create_time_signature_graph(data1, data2, ylabel, title_suffix, filename_suffix):
        fig, ax = plt.subplots(figsize=(15, 8))
        rects1 = ax.bar(x + 0.5 - width/2, data1, width, label='Arrivals', color='blue', alpha=0.7)
        rects2 = ax.bar(x + 0.5 + width/2, data2, width, label='Departures', color='red', alpha=0.7)
        
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel('Hour of Day', fontsize=16)
        ax.set_title(f'{title_suffix} Arrivals to and Departures from Zone {focus_zone} Over the Day', fontsize=24)
        ax.set_xticks(range(24))
        ax.set_xticklabels(hours)
        ax.legend(fontsize=16)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.set_xlim(-0.5, 23.5)
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        output_path = f'/Users/noamgal/Downloads/NUR/celular1819_v1.3/time_signature_zone_{focus_zone}_{filename_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Time signature graph ({filename_suffix}) saved to: {output_path}")
    
    # Create graph for total trips
    create_time_signature_graph(arrivals.values, departures.values, 'Number of Trips', 'Total', 'total')
    
    # Create graph for percentage of trips
    arrivals_percent = (arrivals / arrivals.sum()) * 100
    departures_percent = (departures / departures.sum()) * 100
    create_time_signature_graph(arrivals_percent.values, departures_percent.values, 'Percentage of Trips', 'Percentage of', 'percent')


    
def plot_focus_zone_map(focus_zone):
    # Filter for the specified zone
    filtered_zones = zones[zones['TAZ_1270'] == focus_zone]
    
    # Create the map
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the filtered zone
    filtered_zones.plot(ax=ax, alpha=0.5, edgecolor='black', color='blue')
    
    # Add labels based on 'TAZ_1270'
    for idx, row in filtered_zones.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(text=row['TAZ_1270'], xy=(centroid.x, centroid.y),
                    xytext=(3, 3), textcoords="offset points", fontsize=12,
                    ha='center', va='center', fontweight='bold')
    
    # Zoom to fit the filtered zone with more context
    bounds = filtered_zones.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    ax.set_xlim(bounds[0] - width, bounds[2] + width)
    ax.set_ylim(bounds[1] - height, bounds[3] + height)
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Add map components
    ax.set_title(f'Focus Zone: {focus_zone}', fontsize=24)
    ax.set_axis_off()
    
    # Add scale bar
    ax.add_artist(ScaleBar(dx=1, units="m", location="lower right"))
    
    # Add north arrow
    x, y, arrow_length = 0.05, 0.95, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)
    
    plt.tight_layout()
    output_path = f'/Users/noamgal/Downloads/NUR/celular1819_v1.3/focus_zone_{focus_zone}_map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Focus zone map saved to: {output_path}")
    

def estimate_district_population(df_weekday, df_weekday_arrival, focus_zone):
    to_focus = df_weekday_arrival[df_weekday_arrival['ToZone'] == focus_zone]
    from_focus = df_weekday[df_weekday['fromZone'] == focus_zone]
    
    time_columns = [f'h{i}' for i in range(6, 24)]  # From h6 to h23
    arrivals = to_focus[time_columns].sum()
    departures = from_focus[time_columns].sum()
    
    total_arrivals = arrivals.sum()  # Total arrivals during the period
    
    population = [0]  # Starting population at h6 is 0
    for hour in range(18):  # 18 hours from 6 to 23
        net_change = arrivals.iloc[hour] - departures.iloc[hour]
        new_pop = max(0, population[-1] + net_change)  # Ensure population doesn't go negative
        population.append(new_pop)
    
    population = population[1:]  # Remove initial 0
    population_percent = [(pop / total_arrivals) * 100 for pop in population]
    
    hours = list(range(6, 24))
    
    # Function to create and save a graph
    def create_graph(y_data, y_label, title_suffix, filename_suffix):
        plt.figure(figsize=(15, 8))
        plt.plot(hours, y_data, marker='o')
        plt.title(f'Estimated Population in Zone {focus_zone} {title_suffix} (6 AM to 11 PM)', fontsize=24)
        plt.xlabel('Hour of Day', fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.xticks(hours)
        plt.ylim(0, max(y_data) * 1.1)  # Set y-axis limit to 110% of max value
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate the maximum and minimum population points
        max_pop = max(y_data)
        min_pop = min(y_data)
        max_hour = hours[y_data.index(max_pop)]
        min_hour = hours[y_data.index(min_pop)]
        
        annotation_format = '.1f' if 'percent' in filename_suffix else '.0f'
        percent_sign = '%' if 'percent' in filename_suffix else ''
        
        plt.annotate(f'Max: {max_pop:{annotation_format}}{percent_sign}', xy=(max_hour, max_pop), xytext=(5, 5), 
                     textcoords='offset points', fontsize=12, fontweight='bold')
        plt.annotate(f'Min: {min_pop:{annotation_format}}{percent_sign}', xy=(min_hour, min_pop), xytext=(5, 5), 
                     textcoords='offset points', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = f'/Users/noamgal/Downloads/NUR/celular1819_v1.3/estimated_population_{filename_suffix}_zone_{focus_zone}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Estimated population {filename_suffix} graph saved to: {output_path}")
    
    # Create percentage graph
    create_graph(population_percent, 'Estimated Population (% of Total Arrivals)', 'as % of Total Arrivals', 'percent')
    
    # Create absolute numbers graph
    create_graph(population, 'Estimated Population', '(Absolute Numbers)', 'absolute')
    
    return population, population_percent  # Return both absolute numbers and percentages


def generate_filtered_map(focus_zone):
    print(f"Generating filtered map for focus zone: {focus_zone}")
    
    # Prepare data
    population = population_df.set_index('TAZ_1270')[2019]
    mean_population = population[population > 0].mean()
    population = population.replace(0, mean_population)
    
    trips_to_focus = df_weekday[df_weekday['ToZone'] == focus_zone].groupby('fromZone')[['h' + str(i) for i in range(24)]].sum().sum(axis=1)
    
    # Merge population and trip data with zones
    zones_data = zones.merge(
        pd.DataFrame({
            'TAZ_1270': population.index, 
            'population': population,
            'total_trips': trips_to_focus
        }), 
        on='TAZ_1270', 
        how='left'
    )
    zones_data['total_trips'] = zones_data['total_trips'].fillna(0)
    zones_data['population'] = zones_data['population'].fillna(mean_population)
    zones_data['trips_per_10k'] = (zones_data['total_trips'] / zones_data['population']) * 10000
    
    # Filter zones
    filtered_zones = zones_data[
        (zones_data['trips_per_10k'] > 10) & 
        (zones_data['trips_per_10k'].notna()) &
        (zones_data['TAZ_1270'] != focus_zone) &
        (zones_data['population'] >= 25)  # New condition
    ]
    
    # Create Folium map
    focus_zone_geom = zones[zones['TAZ_1270'] == focus_zone].to_crs(epsg=4326)
    focus_zone_center = focus_zone_geom.geometry.centroid.iloc[0]
    m = folium.Map(location=[focus_zone_center.y, focus_zone_center.x], zoom_start=9)

    # Set up color map
    vmin = 10
    vmax = np.percentile(filtered_zones['trips_per_10k'], 95)  # 95th percentile for capping
    colormap = LinearColormap(colors=['#FEE391', '#D9F0A3', '#ADDD8E', '#78C679', '#31A354'], vmin=vmin, vmax=vmax)
    
    def style_function(feature):
        trips = feature['properties']['trips_per_10k']
        if trips > vmax:
            return {'fillColor': '#006400', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
        return {'fillColor': colormap(trips), 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}

    # Add GeoJson layer
    folium.GeoJson(
        filtered_zones.to_crs(epsg=4326),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['TAZ_1270', 'population', 'total_trips', 'trips_per_10k'],
            aliases=['TAZ_1270:', 'Population:', 'Total Trips:', 'Trips per 10k:'],
            labels=True,
            sticky=False,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    # Highlight focus zone
    folium.GeoJson(focus_zone_geom, style_function=lambda x: {'color': 'red', 'weight': 3, 'fillOpacity': 0}).add_to(m)

    # Add a star marker for the focus zone
    folium.Marker(
        location=[focus_zone_center.y, focus_zone_center.x],
        icon=folium.Icon(color='red', icon='star'),
    ).add_to(m)

    # Add the colormap to the map
    colormap.add_to(m)

    # Add a custom legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                width: 250px; height: 120px; 
                background-color: white; 
                border-radius: 5px; 
                padding: 10px; 
                z-index: 9999;">
        <p style="font-size:14px; margin-bottom: 5px;"><b>Legend</b></p>
        <p style="font-size:12px; margin-top: 0;">Trips per 10,000 people</p>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="background: linear-gradient(to right, #FEE391, #D9F0A3, #ADDD8E, #78C679, #31A354); 
                        width: 150px; height: 20px;"></div>
            <div style="display: flex; justify-content: space-between; width: 150px;">
                <span style="font-size: 10px;">10</span>
                <span style="font-size: 10px;">{vmax:.0f}</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="background-color: #006400; width: 20px; height: 20px;"></div>
            <span style="font-size: 10px; margin-left: 5px;">> {vmax:.0f} (95th percentile)</span>
        </div>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))

    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 800px; 
                background-color: white; 
                border-radius: 5px; 
                padding: 10px; 
                z-index: 9999;">
        <h3 align="center" style="font-size:16px;">
            <b>Filtered Map: Trips to Focus Zone ({focus_zone})</b>
        </h3>
        <p style="font-size:12px;">
            Showing only TAZ_1270 zones with more than 10 trips per 10,000 people to the focus zone and population ≥ 25.
            Zones with trips > {vmax:.0f} per 10k (95th percentile) are colored dark green.
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save Folium map
    output_path = f'/Users/noamgal/Downloads/NUR/celular1819_v1.3/focus_zone_{focus_zone}_filtered_map.html'
    m.save(output_path)
    print(f"Filtered map saved to: {output_path}")

# The main execution block remains the same
if __name__ == "__main__":
    focus_zone = int(input("Enter the TAZ code for the focus zone: "))
    generate_maps_and_graphs(focus_zone)
    generate_filtered_map(focus_zone)


def compare_two_zones_time_signature(df_weekday, df_weekday_arrival, focus_zone1, focus_zone2, zone1_name, zone2_name):
    # Prepare data for both zones
    def prepare_zone_data(zone):
        to_zone = df_weekday_arrival[df_weekday_arrival['ToZone'] == zone]
        from_zone = df_weekday[df_weekday['fromZone'] == zone]
        
        time_columns = [f'h{i}' for i in range(24)]
        arrivals = to_zone[time_columns].sum()
        departures = from_zone[time_columns].sum()
        
        arrivals_percent = (arrivals / arrivals.sum()) * 100
        departures_percent = (departures / departures.sum()) * 100
        
        # Reset index to numeric values
        arrivals_percent.index = range(24)
        departures_percent.index = range(24)
        
        return arrivals_percent, departures_percent

    arrivals1, departures1 = prepare_zone_data(focus_zone1)
    arrivals2, departures2 = prepare_zone_data(focus_zone2)

    # Set up the plot
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")
    
    hours = range(24)

    # Plot lines with markers
    plt.plot(hours, arrivals1, color='#1f77b4', linestyle='-', linewidth=2, marker='o', markersize=6, label=f'{zone1_name} Arrivals')
    plt.plot(hours, departures1, color='#1f77b4', linestyle='--', linewidth=2, marker='s', markersize=6, label=f'{zone1_name} Departures')
    plt.plot(hours, arrivals2, color='#ff7f0e', linestyle='-', linewidth=2, marker='o', markersize=6, label=f'{zone2_name} Arrivals')
    plt.plot(hours, departures2, color='#ff7f0e', linestyle='--', linewidth=2, marker='s', markersize=6, label=f'{zone2_name} Departures')

    # Customize the plot
    plt.title('Comparison of Arrivals and Departures: Beer Sheva ID vs Matam Haifa', fontsize=24, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=18)
    plt.ylabel('Percentage of Daily Trips', fontsize=18)
    plt.xticks(hours, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, max(arrivals1.max(), departures1.max(), arrivals2.max(), departures2.max()) * 1.1)

    # Add legend
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for peak hours
    def annotate_peak(data, color, is_arrival):
        peak_hour = data.idxmax()
        peak_value = data.max()
        plt.annotate(f'Peak: {peak_value:.1f}%',
                     xy=(peak_hour, peak_value), xytext=(5, 5),
                     textcoords='offset points', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
                     fontsize=12, fontweight='bold', color=color)

    annotate_peak(arrivals1, '#1f77b4', True)
    annotate_peak(departures1, '#1f77b4', False)
    annotate_peak(arrivals2, '#ff7f0e', True)
    annotate_peak(departures2, '#ff7f0e', False)

    # Adjust layout and save
    plt.tight_layout()
    output_path = f'/Users/noamgal/Downloads/NUR/celular1819_v1.3/comparison_{focus_zone1}_vs_{focus_zone2}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison graph saved to: {output_path}")

# Usage
focus_zone1 = 101104  # Beer Sheva ID
focus_zone2 = 100694  # Matam Haifa
compare_two_zones_time_signature(df_weekday, df_weekday_arrival, focus_zone1, focus_zone2, "Beer Sheva ID", "Matam Haifa")
