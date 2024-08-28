import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
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

# Preprocessing to filter out problematic TAZs
valid_tazs = df_weekday['ToZone'].unique()
valid_tazs = [taz for taz in valid_tazs if taz in zones['TAZ_1270'].values]
valid_tazs = [taz for taz in valid_tazs if taz in population_df['TAZ_1270'].values]

print("Data loaded successfully")

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Mobility Data Dashboard"),
    dcc.Dropdown(
        id='taz-dropdown',
        options=[{'label': str(i), 'value': i} for i in valid_tazs],
        value=valid_tazs[0],
        style={'width': '50%'}
    ),
    dcc.Graph(id='geopandas-map'),
    dcc.Graph(id='time-signature'),
    dcc.Graph(id='trips-by-distance'),
    html.Div(id='output-message')
])

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

    # Create histogram with 5 km increments
    bins = np.arange(0, max_distance + 5, 5)
    hist, bin_edges = np.histogram(filtered_trips['distance'], bins=bins, weights=filtered_trips['total_trips'])
    
    # Calculate cumulative percentage
    cumulative = np.cumsum(hist)
    cumulative_percent = cumulative / cumulative[-1] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Histogram
    fig.add_trace(
        go.Bar(x=bin_edges[:-1], y=hist, name='Number of Trips'),
        secondary_y=False,
    )

    # Cumulative percentage line
    fig.add_trace(
        go.Scatter(x=bin_edges[:-1], y=cumulative_percent, mode='lines', name='Cumulative Percentage'),
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

    print("Trips by distance plot created successfully")
    return fig

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
    zones_data['trips_per_10k'] = zones_data['trips_per_10k'].fillna(0)
    zones_data['total_trips'] = zones_data['total_trips'].fillna(0)

    # Calculate total arrivals and departures for focus zone
    total_arrivals = df_weekday_arrival[df_weekday_arrival['ToZone'] == focus_zone][['h' + str(i) for i in range(24)]].sum().sum()
    total_departures = df_weekday[df_weekday['fromZone'] == focus_zone][['h' + str(i) for i in range(24)]].sum().sum()

    # Set up color map
    vmin = 10  # Minimum value for coloring (less than this will be transparent)
    vmax = np.percentile(zones_data['trips_per_10k'][zones_data['trips_per_10k'] > 10], 95)  # 95th percentile for capping

    # Create a custom colorscale (from light blue to dark blue)
    colorscale = [
        [0, 'rgba(0,0,0,0)'],  # Fully transparent for values less than vmin
        [0.1, 'rgba(240,249,255,0.8)'],  # Lightest blue
        [0.25, 'rgba(204,224,255,0.8)'],
        [0.5, 'rgba(102,169,255,0.8)'],
        [0.75, 'rgba(0,112,255,0.8)'],
        [1, 'rgba(0,41,117,0.8)']  # Darkest blue
    ]

    # Convert to WGS84 for Plotly
    zones_data = zones_data.to_crs(epsg=4326)

    # Get focus zone geometry and calculate bounding box
    focus_zone_geo = zones_data[zones_data['TAZ_1270'] == focus_zone]
    bbox = focus_zone_geo.total_bounds
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    # Calculate zoom level based on bounding box
    zoom = min(12, max(8, 11 - np.log2(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 0.005)))  # Adjust constants as needed

    # Create Plotly figure
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "mapbox"}]])

    # Add choropleth layer
    choropleth = go.Choroplethmapbox(
        geojson=zones_data.geometry.__geo_interface__,
        locations=zones_data.index,
        z=zones_data['trips_per_10k'],
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        marker_opacity=1,
        marker_line_width=0,
        colorbar_title='Trips per 10k people',
        hovertemplate="<br>".join([
            "TAZ: %{customdata[0]}",
            "Population: %{customdata[1]:,.0f}",
            "Total Trips: %{customdata[2]:,.0f}",
            "Trips per 10k: %{customdata[3]:,.2f}"
        ]),
        customdata=zones_data[['TAZ_1270', 'population', 'total_trips', 'trips_per_10k']]
    )
    fig.add_trace(choropleth)

    # Add focus zone as a separate layer
    focus_color = '#FF4136'  # A bright red color for contrast
    focus_layer = go.Choroplethmapbox(
        geojson=focus_zone_geo.geometry.__geo_interface__,
        locations=focus_zone_geo.index,
        z=[1],
        colorscale=[[0, focus_color], [1, focus_color]],
        marker_opacity=0.8,
        marker_line_width=2,
        showscale=False,
        hovertemplate="<b>Focus Zone</b><br>" +
                      f"TAZ: {focus_zone}<br>" +
                      f"Total Arrivals: {total_arrivals:,.0f}<br>" +
                      f"Total Departures: {total_departures:,.0f}<extra></extra>"
    )
    fig.add_trace(focus_layer)

    # Update layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600,
        title=f'Trips per 10,000 people to Zone {focus_zone}'
    )

    # Add scale bar
    scale_bar_color = 'black'
    fig.add_shape(type="line",
        x0=0.01, y0=0.05, x1=0.11, y1=0.05,
        line=dict(color=scale_bar_color, width=3),
        xref="paper", yref="paper"
    )
    for i, label in enumerate(['0', '5', '10']):
        fig.add_annotation(
            x=0.01 + i * 0.05, y=0.04,
            text=label,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=10, color=scale_bar_color)
        )
    fig.add_annotation(
        x=0.06, y=0.07,
        text="km",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=10, color=scale_bar_color)
    )

    # Add north arrow
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="↑",
        showarrow=False,
        font=dict(size=24, color=scale_bar_color),
    )
    fig.add_annotation(
        x=0.02,
        y=0.95,
        xref="paper",
        yref="paper",
        text="N",
        showarrow=False,
        font=dict(size=16, color=scale_bar_color),
    )

    print("Geopandas map created successfully")
    return fig
@app.callback(
    [Output('geopandas-map', 'figure'),
     Output('time-signature', 'figure'),
     Output('trips-by-distance', 'figure'),
     Output('output-message', 'children'),
     Output('taz-dropdown', 'value')],
    [Input('taz-dropdown', 'value'),
     Input('geopandas-map', 'clickData')],
    [State('taz-dropdown', 'value')]
)
def update_graphs(selected_taz, clickData, current_taz):
    ctx = dash.callback_context
    if not ctx.triggered:
        taz = current_taz
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if input_id == 'geopandas-map':
            if clickData is not None and 'customdata' in clickData['points'][0]:
                taz = clickData['points'][0]['customdata'][0]
            else:
                taz = current_taz
        else:
            taz = selected_taz

    print(f"Updating graphs for TAZ: {taz}")
    try:
        geopandas_map = create_geopandas_map(taz)
        time_signature_fig = plot_time_signature(df_weekday, df_weekday_arrival, taz)
        trips_by_distance_fig = plot_trips_by_distance(taz)

        message = f"All graphs updated successfully for TAZ {taz}."
        print(message)
        return geopandas_map, time_signature_fig, trips_by_distance_fig, message, taz
    except Exception as e:
        error_message = f"Error updating graphs for TAZ {taz}: {str(e)}"
        print(error_message)
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=error_message,
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, empty_fig, error_message, current_taz

# Run the app
if __name__ == '__main__':
    print("Starting the Dash app...")
    app.run_server(debug=True)

print("Dashboard is running on http://127.0.0.1:8050/")
