import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import numpy as np

# Load preprocessed data
preprocessed_data = pd.read_csv('/Users/noamgal/Downloads/NUR/celular1819_v1.3/preprocessed_mobility_data.csv')
zones = gpd.read_file('/Users/noamgal/Downloads/NUR/celular1819_v1.3/zones_3857.geojson')

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Mobility Data Dashboard"),
    dcc.Dropdown(
        id='taz-dropdown',
        options=[{'label': str(i), 'value': i} for i in preprocessed_data['ToZone'].unique()],
        value=preprocessed_data['ToZone'].min(),
        style={'width': '50%'}
    ),
    dcc.Graph(id='map-output'),
    dcc.Graph(id='trips-by-distance'),
    dcc.Graph(id='time-signature'),
    dcc.Graph(id='estimated-population')
])

def create_map(focus_zone):
    # Filter data for the focus zone
    zone_data = preprocessed_data[preprocessed_data['ToZone'] == focus_zone]
    
    # Merge with zones geodataframe
    gdf = zones.merge(zone_data, left_on='TAZ_1270', right_on='fromZone')
    
    # Create the map
    fig = px.choropleth_mapbox(gdf, 
                               geojson=gdf.geometry.__geo_interface__, 
                               locations=gdf.index, 
                               color='trips_per_10k',
                               color_continuous_scale="Viridis",
                               mapbox_style="carto-positron",
                               zoom=9, 
                               center={"lat": gdf.geometry.centroid.y.mean(), 
                                       "lon": gdf.geometry.centroid.x.mean()},
                               opacity=0.5,
                               labels={'trips_per_10k':'Trips per 10k people'})

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    return fig

def plot_trips_by_distance(focus_zone):
    zone_data = preprocessed_data[preprocessed_data['ToZone'] == focus_zone]
    
    fig = px.histogram(zone_data, x='distance', y='total_trips', 
                       nbins=30, range_x=[0, 150],
                       labels={'distance': 'Distance (km)', 'total_trips': 'Number of Trips'},
                       title=f'Histogram of Trips to Zone {focus_zone} by Distance')
    
    return fig

def plot_time_signature(focus_zone):
    zone_data = preprocessed_data[preprocessed_data['ToZone'] == focus_zone]
    
    hours = list(range(24))
    arrivals = [zone_data[f'h{i}'].sum() for i in hours]
    departures = preprocessed_data[preprocessed_data['fromZone'] == focus_zone][[f'h{i}' for i in hours]].sum()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=arrivals, name='Arrivals', marker_color='blue'))
    fig.add_trace(go.Bar(x=hours, y=departures, name='Departures', marker_color='red'))
    
    fig.update_layout(
        title=f'Arrivals to and Departures from Zone {focus_zone} Over the Day',
        xaxis_title='Hour of Day',
        yaxis_title='Number of Trips',
        barmode='group'
    )
    
    return fig

def estimate_district_population(focus_zone):
    zone_data = preprocessed_data[preprocessed_data['ToZone'] == focus_zone]
    
    hours = list(range(6, 24))
    arrivals = [zone_data[f'h{i}'].sum() for i in hours]
    departures = preprocessed_data[preprocessed_data['fromZone'] == focus_zone][[f'h{i}' for i in hours]].sum()
    
    population = [0]
    for arrival, departure in zip(arrivals, departures):
        net_change = arrival - departure
        new_pop = max(0, population[-1] + net_change)
        population.append(new_pop)
    
    population = population[1:]
    total_arrivals = sum(arrivals)
    population_percent = [(pop / total_arrivals) * 100 for pop in population]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=population, mode='lines+markers', name='Estimated Population'))
    fig.add_trace(go.Scatter(x=hours, y=population_percent, mode='lines+markers', name='Population (% of Total Arrivals)', yaxis='y2'))
    
    fig.update_layout(
        title=f'Estimated Population in Zone {focus_zone} (6 AM to 11 PM)',
        xaxis_title='Hour of Day',
        yaxis_title='Estimated Population',
        yaxis2=dict(title='Population (% of Total Arrivals)', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='Black', borderwidth=1)
    )
    
    return fig

# Callback to update all graphs
@app.callback(
    [Output('map-output', 'figure'),
     Output('trips-by-distance', 'figure'),
     Output('time-signature', 'figure'),
     Output('estimated-population', 'figure')],
    [Input('taz-dropdown', 'value')]
)
def update_graphs(selected_taz):
    try:
        map_fig = create_map(selected_taz)
        trips_by_distance_fig = plot_trips_by_distance(selected_taz)
        time_signature_fig = plot_time_signature(selected_taz)
        estimated_population_fig = estimate_district_population(selected_taz)
        
        return map_fig, trips_by_distance_fig, time_signature_fig, estimated_population_fig
    except Exception as e:
        print(f"Error updating graphs for TAZ {selected_taz}: {str(e)}")
        # Return empty figures if there's an error
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error: {str(e)}",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, empty_fig, empty_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

print("Dashboard is running on http://127.0.0.1:8050/")