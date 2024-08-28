import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import geopandas as gpd
import numpy as np

print("Loading data...")
# Load data
df_weekday = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_v1.csv')
df_weekday_arrival = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/AvgDayHourlyTrips201819_1270_weekday_arrival_v1.2.csv')
zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp')
population_df = pd.read_excel('/Users/noamgal/Downloads/NUR/celular1819_v1.3/1270_population.xlsx')

# Convert zones to Web Mercator projection
zones = zones.to_crs(epsg=3857)

print("Data loaded successfully")

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Mobility Data Dashboard"),
    dcc.Dropdown(
        id='taz-dropdown',
        options=[{'label': str(i), 'value': i} for i in df_weekday['ToZone'].unique()],
        value=df_weekday['ToZone'].min(),
        style={'width': '50%'}
    ),
    dcc.Graph(id='geopandas-map'),
    dcc.Graph(id='time-signature'),
    dcc.Graph(id='trips-by-distance'),
    dcc.Graph(id='estimated-population'),
    dcc.Graph(id='per-capita-trips-map'),
    html.Div(id='output-message')
])
    
def create_geopandas_map(focus_zone):
    print(f"Creating geopandas map for focus zone: {focus_zone}")

    # Calculate trips to focus zone
    trips_to_focus = df_weekday[df_weekday['ToZone'] == focus_zone].groupby('fromZone')[['h' + str(i) for i in range(24)]].sum().sum(axis=1)

    # Prepare population data
    population = population_df.set_index('TAZ_1270')[2019]
    mean_population = population[population > 0].mean()
    population = population.replace(0, mean_population)

    # Calculate trips per 10,000 people
    trips_per_10k = (trips_to_focus / population) * 10000

    # Create a DataFrame with all the data
    mapping_data = pd.DataFrame({
        'TAZ_1270': population.index,
        'trips_per_10k': trips_per_10k,
        'total_trips': trips_to_focus,
        'population': population
    })

    # Merge with zones
    zones_data = zones.merge(mapping_data, on='TAZ_1270', how='left')

    print(f"Shape of zones_data: {zones_data.shape}")
    print(f"Columns in zones_data: {zones_data.columns.tolist()}")
    print(f"Sample of zones_data:\n{zones_data.head()}")

    # Ensure zones_data is in EPSG:3857 (Web Mercator) projection
    zones_data = zones_data.to_crs(epsg=3857)

    # Convert to WGS84 for Plotly
    zones_data = zones_data.to_crs(epsg=4326)

    # Create Plotly figure
    fig = px.choropleth_mapbox(zones_data, 
                               geojson=zones_data.geometry.__geo_interface__, 
                               locations=zones_data.index, 
                               color='trips_per_10k',
                               color_continuous_scale="Viridis",
                               mapbox_style="open-street-map",
                               zoom=7, 
                               center={"lat": zones_data.geometry.centroid.y.mean(), 
                                       "lon": zones_data.geometry.centroid.x.mean()},
                               opacity=0.5,
                               labels={'trips_per_10k':'Trips per 10k people'})

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600,
                      title=f'Trips per 10,000 people to Zone {focus_zone}')

    print("Geopandas map created successfully")
    return fig

def plot_time_signature(df_weekday, df_weekday_arrival, focus_zone):
    print(f"Plotting time signature for focus zone: {focus_zone}")
    to_focus = df_weekday_arrival[df_weekday_arrival['ToZone'] == focus_zone]
    from_focus = df_weekday[df_weekday['fromZone'] == focus_zone]

    time_columns = [f'h{i}' for i in range(24)]
    arrivals = to_focus[time_columns].sum()
    departures = from_focus[time_columns].sum()

    arrivals_percent = (arrivals / arrivals.sum()) * 100
    departures_percent = (departures / departures.sum()) * 100

    print(f"Shape of arrivals_percent: {arrivals_percent.shape}")
    print(f"Shape of departures_percent: {departures_percent.shape}")

    hours = list(range(24))  # Convert range to list

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=arrivals_percent, name='Arrivals', marker_color='blue'))
    fig.add_trace(go.Bar(x=hours, y=departures_percent, name='Departures', marker_color='red'))

    fig.update_layout(
        title=f'Percentage of Arrivals to and Departures from Zone {focus_zone} Over the Day',
        xaxis_title='Hour of Day',
        yaxis_title='Percentage of Trips',
        barmode='group',
        legend=dict(x=0.7, y=1, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    print("Time signature plot created successfully")
    return fig

def plot_trips_by_distance(focus_zone):
    print(f"Plotting trips by distance for focus zone: {focus_zone}")
    
    # Calculate distances
    focus_point = zones[zones['TAZ_1270'] == focus_zone].geometry.centroid.iloc[0]
    zones['distance'] = zones.geometry.centroid.distance(focus_point) / 1000  # Convert to km
    
    # Get trips data
    trips_to_focus = df_weekday[df_weekday['ToZone'] == focus_zone].groupby('fromZone')[['h' + str(i) for i in range(24)]].sum().sum(axis=1)
    
    # Merge trips and distance data
    trips_with_distance = zones.merge(trips_to_focus.reset_index(), left_on='TAZ_1270', right_on='fromZone', how='right')
    trips_with_distance = trips_with_distance.rename(columns={0: 'total_trips'})
    
    # Filter and sort data
    max_distance = 150
    filtered_trips = trips_with_distance[trips_with_distance['distance'] <= max_distance].sort_values('distance')
    
    # Calculate cumulative percentage
    filtered_trips['cumulative_trips'] = filtered_trips['total_trips'].cumsum()
    filtered_trips['cumulative_percent'] = (filtered_trips['cumulative_trips'] / filtered_trips['total_trips'].sum()) * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Histogram
    fig.add_trace(
        go.Bar(x=filtered_trips['distance'], y=filtered_trips['total_trips'], name='Number of Trips'),
        secondary_y=False,
    )
    
    # Cumulative percentage line
    fig.add_trace(
        go.Scatter(x=filtered_trips['distance'], y=filtered_trips['cumulative_percent'], 
                   mode='lines', name='Cumulative Percentage'),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=f'Histogram of Trips to Zone {focus_zone} by Distance',
        xaxis_title='Distance (km)',
        yaxis_title='Number of Trips',
        legend=dict(x=0.7, y=1, bgcolor='rgba(255, 255, 255, 0.8)')
    )
    
    fig.update_yaxes(title_text="Number of Trips", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True)
    print(f"Shape of filtered_trips: {filtered_trips.shape}")
    print(f"Columns in filtered_trips: {filtered_trips.columns.tolist()}")
    print(f"Sample of filtered_trips:\n{filtered_trips.head()}")
    print("Trips by distance plot created successfully")
    return fig
        
def estimate_district_population(df_weekday, df_weekday_arrival, focus_zone):
    print(f"Estimating district population for focus zone: {focus_zone}")
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=population, mode='lines+markers', name='Estimated Population'))
    fig.add_trace(go.Scatter(x=hours, y=population_percent, mode='lines+markers', name='Population (% of Total Arrivals)', yaxis='y2'))

    fig.update_layout(
        title=f'Estimated Population in Zone {focus_zone} (6 AM to 11 PM)',
        xaxis_title='Hour of Day',
        yaxis_title='Estimated Population',
        yaxis2=dict(title='Population (% of Total Arrivals)', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    print(f"Shape of population: {len(population)}")
    print(f"Shape of population_percent: {len(population_percent)}")
    print("Estimated population plot created successfully")
    return fig

def create_per_capita_trips_map(focus_zone, buffer_distance_km=40):
    print(f"Creating Per Capita Trips map for focus zone: {focus_zone}")

    # Calculate trips to focus zone
    trips_to_focus = df_weekday[(df_weekday['ToZone'] == focus_zone) & (df_weekday['fromZone'] != focus_zone)].groupby('fromZone')[['h' + str(i) for i in range(24)]].sum().sum(axis=1)

    # Prepare population data
    population = population_df.set_index('TAZ_1270')[2019]
    mean_population = population[population > 0].mean()
    population = population.replace(0, mean_population)

    # Calculate trips per 10,000 people
    trips_per_10k = (trips_to_focus / population) * 10000

    # Create a DataFrame with all the data
    mapping_data = pd.DataFrame({
        'TAZ_1270': population.index,
        'trips_per_10k': trips_per_10k,
        'total_trips': trips_to_focus,
        'population': population
    })

    # Merge with zones
    zones_data = zones.merge(mapping_data, on='TAZ_1270', how='left')

    # Get the focus zone
    focus_zone_geo = zones_data[zones_data['TAZ_1270'] == focus_zone]

    # Create a buffer around the focus zone
    buffer = focus_zone_geo.to_crs(epsg=3857).buffer(buffer_distance_km * 1000)  # Convert km to meters

    # Select zones that intersect with the buffer, excluding the focus zone
    zones_within_buffer = zones_data[(zones_data.to_crs(epsg=3857).intersects(buffer.iloc[0])) & (zones_data['TAZ_1270'] != focus_zone)]

    # Convert to WGS84 for Plotly
    zones_within_buffer = zones_within_buffer.to_crs(epsg=4326)

    # Create Plotly figure
    fig = px.choropleth_mapbox(zones_within_buffer, 
                               geojson=zones_within_buffer.geometry.__geo_interface__, 
                               locations=zones_within_buffer.index, 
                               color='trips_per_10k',
                               color_continuous_scale="Viridis",
                               mapbox_style="open-street-map",
                               zoom=9, 
                               center={"lat": focus_zone_geo.to_crs(epsg=4326).geometry.centroid.y.iloc[0], 
                                       "lon": focus_zone_geo.to_crs(epsg=4326).geometry.centroid.x.iloc[0]},
                               opacity=0.5,
                               labels={'trips_per_10k':'Trips per 10k people'})

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600,
                      title=f'Per Capita Trips to Focus Zone {focus_zone} within {buffer_distance_km}km')

    print("Per Capita Trips map created successfully")
    return fig

@app.callback(
    [Output('geopandas-map', 'figure'),
     Output('time-signature', 'figure'),
     Output('trips-by-distance', 'figure'),
     Output('estimated-population', 'figure'),
     Output('per-capita-trips-map', 'figure'),
     Output('output-message', 'children')],
    [Input('taz-dropdown', 'value')]
)
def update_graphs(selected_taz):
    print(f"Updating graphs for TAZ: {selected_taz}")
    try:
        geopandas_map = create_geopandas_map(selected_taz)
        time_signature_fig = plot_time_signature(df_weekday, df_weekday_arrival, selected_taz)
        trips_by_distance_fig = plot_trips_by_distance(selected_taz)
        estimated_population_fig = estimate_district_population(df_weekday, df_weekday_arrival, selected_taz)
        per_capita_trips_map = create_per_capita_trips_map(selected_taz)

        message = f"All graphs updated successfully for TAZ {selected_taz}."
        print(message)
        return geopandas_map, time_signature_fig, trips_by_distance_fig, estimated_population_fig, per_capita_trips_map, message
    except Exception as e:
        error_message = f"Error updating graphs for TAZ {selected_taz}: {str(e)}"
        print(error_message)
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=error_message,
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, error_message
    
# Run the app
if __name__ == '__main__':
    print("Starting the Dash app...")
    app.run_server(debug=True)

print("Dashboard is running on http://127.0.0.1:8050/")