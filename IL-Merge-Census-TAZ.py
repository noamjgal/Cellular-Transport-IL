import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import fiona

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
TAZ_SHAPEFILE = '/Users/noamgal/Downloads/NUR/celular1819_v1.3/Shape_files/1270_02.09.2021.shp'
CENSUS_GDB = '/Users/noamgal/Downloads/NUR/IL-Census-2022/census_2022_statistical_areas_2022.gdb'
OUTPUT_CSV = '/Users/noamgal/Downloads/NUR/taz_census_estimate_spatial.csv'

# Define economic statistics to include and their aggregation methods
ECONOMIC_STATS = {
    'sexRatio': 'mean',
    'inst_pcnt': 'weighted_mean',
    'Foreign_pcnt': 'weighted_mean',
    'age0_19_pcnt': 'weighted_mean',
    'age20_64_pcnt': 'weighted_mean',
    'age65_pcnt': 'weighted_mean',
    'DependencyRatio': 'weighted_mean',
    'age_median': 'weighted_median',
    'm_age_median': 'weighted_median',
    'w_age_median': 'weighted_median',
    'married18_34_pcnt': 'weighted_mean',
    'married45_54_pcnt': 'weighted_mean',
    'j_isr_pcnt': 'weighted_mean',
    'j_abr_pcnt': 'weighted_mean',
    'aliya2002_pcnt': 'weighted_mean',
    'aliya2010_pcnt': 'weighted_mean',
    'israel_pcnt': 'weighted_mean',
    'asia_pcnt': 'weighted_mean',
    'africa_pcnt': 'weighted_mean',
    'europe_pcnt': 'weighted_mean',
    'america_pcnt': 'weighted_mean',
    'MarriageAge_mdn': 'weighted_median',
    'm_MarriageAge_mdn': 'weighted_median',
    'w_MarriageAge_mdn': 'weighted_median',
    'ChldBorn_avg': 'weighted_mean',
    'koshi5_pcnt': 'weighted_mean',
    'koshi65_pcnt': 'weighted_mean',
    'AcadmCert_pcnt': 'weighted_mean',
    'WrkY_pcnt': 'weighted_mean',
    'Empl_pcnt': 'weighted_mean',
    'SelfEmpl_pcnt': 'weighted_mean',
    'HrsWrkWk_avg': 'weighted_mean',
    'Wrk_15_17_pcnt': 'weighted_mean',
    'WrkOutLoc_pcnt': 'weighted_mean',
    'employeesAnnual_medWage': 'weighted_median',
    'EmployeesWage_decile9Up': 'weighted_mean',
    'SelfEmployedAnnual_medWage': 'weighted_median',
    'SelfEmployedWage_decile9Up': 'weighted_mean',
    'size_avg': 'weighted_mean',
    'hh0_5_pcnt': 'weighted_mean',
    'hh18_24_pcnt': 'weighted_mean',
    'Computer_avg': 'weighted_mean',
    'Vehicle0_pcnt': 'weighted_mean',
    'Vehicle2up_pcnt': 'weighted_mean',
    'Parking_pcnt': 'weighted_mean',
    'own_pcnt': 'weighted_mean',
    'rent_pcnt': 'weighted_mean'
}

def load_data() -> tuple:
    """
    Load TAZ and census data.
    """
    logging.info("Loading data...")
    taz_gdf = gpd.read_file(TAZ_SHAPEFILE)
    
    layers = fiona.listlayers(CENSUS_GDB)
    if layers:
        census_layer = layers[0]
        census_gdf = gpd.read_file(CENSUS_GDB, layer=census_layer)
    else:
        raise ValueError("No layers found in the geodatabase")
    
    return taz_gdf, census_gdf

def prepare_data(taz_gdf: gpd.GeoDataFrame, census_gdf: gpd.GeoDataFrame) -> tuple:
    """
    Prepare the data for spatial allocation.
    """
    logging.info("Preparing data...")
    if taz_gdf.crs != census_gdf.crs:
        census_gdf = census_gdf.to_crs(taz_gdf.crs)
    
    taz_gdf = taz_gdf[taz_gdf.geometry.is_valid].copy()
    census_gdf = census_gdf[census_gdf.geometry.is_valid].copy()
    
    taz_gdf['taz_area'] = taz_gdf.geometry.area
    census_gdf['census_area'] = census_gdf.geometry.area
    
    taz_column = 'TAZ' if 'TAZ' in taz_gdf.columns else 'TAZ_1270' if 'TAZ_1270' in taz_gdf.columns else None
    if taz_column is None:
        raise ValueError("Could not find TAZ column in the TAZ GeoDataFrame")
    taz_gdf = taz_gdf.rename(columns={taz_column: 'TAZ'})
    
    # Log summary of original census data
    logging.info("Summary of original census data:")
    for col in ECONOMIC_STATS.keys():
        if col in census_gdf.columns:
            logging.info(f"{col}: min={census_gdf[col].min()}, max={census_gdf[col].max()}, mean={census_gdf[col].mean()}")
        else:
            logging.warning(f"{col} not found in census data")
    
    return taz_gdf, census_gdf

def spatial_allocation(taz_gdf: gpd.GeoDataFrame, census_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Perform spatial allocation of census data to TAZ zones.
    """
    logging.info("Performing spatial allocation...")
    
    joined = gpd.overlay(taz_gdf, census_gdf, how='intersection')
    joined['intersection_area'] = joined.geometry.area
    joined['proportion'] = joined['intersection_area'] / joined['census_area']
    
    joined['allocated_pop'] = joined['proportion'] * joined['pop_approx']
    
    for stat, method in ECONOMIC_STATS.items():
        if stat in joined.columns:
            joined[f'allocated_{stat}'] = joined[stat]
        else:
            logging.warning(f"{stat} not found in joined data, skipping allocation")
    
    return joined

def weighted_average(group, value_column, weight_column):
    if isinstance(group, pd.Series):
        return group.iloc[0]  # For single value series, just return the value
    if value_column not in group.columns:
        logging.warning(f"{value_column} not found in group, returning NaN")
        return np.nan
    return np.average(group[value_column], weights=group[weight_column])

def aggregate_to_taz(allocated_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Aggregate allocated data to TAZ level.
    """
    logging.info("Aggregating data to TAZ level...")
    
    agg_dict = {
        'allocated_pop': 'sum',
        'taz_area': 'first',
    }
    
    for stat, method in ECONOMIC_STATS.items():
        if f'allocated_{stat}' in allocated_gdf.columns:
            if method in ['weighted_mean', 'weighted_median']:
                agg_dict[f'allocated_{stat}'] = lambda x, stat=stat: weighted_average(x, f'allocated_{stat}', 'allocated_pop')
            else:
                agg_dict[f'allocated_{stat}'] = 'mean'
        else:
            logging.warning(f"allocated_{stat} not found in allocated data, skipping aggregation")
    
    taz_data = allocated_gdf.groupby('TAZ').agg(agg_dict)
    
    # Calculate final statistics
    taz_data['population'] = taz_data['allocated_pop']
    taz_data['area'] = taz_data['taz_area']
    taz_data['pop_density'] = taz_data['population'] / taz_data['area']
    
    for stat in ECONOMIC_STATS.keys():
        if f'allocated_{stat}' in taz_data.columns:
            taz_data[stat] = taz_data[f'allocated_{stat}']
            taz_data = taz_data.drop(columns=[f'allocated_{stat}'])
        else:
            logging.warning(f"allocated_{stat} not found in aggregated data, skipping final calculation")
    
    taz_data = taz_data.drop(columns=['allocated_pop', 'taz_area'])
    
    return taz_data

def validate_results(taz_data: pd.DataFrame, census_gdf: gpd.GeoDataFrame):
    """
    Validate the results of the spatial allocation.
    """
    logging.info("Validating results...")
    
    # Check for missing data
    missing_data = taz_data.isnull().sum()
    if missing_data.any():
        logging.warning(f"Columns with missing data:\n{missing_data[missing_data > 0]}")
    
    # Compare total population
    taz_total_pop = taz_data['population'].sum()
    census_total_pop = census_gdf['pop_approx'].sum()
    logging.info(f"Total population - TAZ estimate: {taz_total_pop:.0f}, Census: {census_total_pop:.0f}")
    
    # Check range of percentages and other key statistics
    for col in taz_data.columns:
        if col.endswith('_pcnt'):
            if (taz_data[col] < 0).any() or (taz_data[col] > 100).any():
                logging.warning(f"Invalid percentage range in column: {col}")
        elif col in ['age_median', 'm_age_median', 'w_age_median', 'MarriageAge_mdn', 'm_MarriageAge_mdn', 'w_MarriageAge_mdn']:
            if (taz_data[col] < 0).any() or (taz_data[col] > 120).any():
                logging.warning(f"Suspicious age value in column: {col}")
    
    # Log summary statistics for key columns
    key_columns = ['pop_density', 'age_median', 'sexRatio', 'DependencyRatio'] + [col for col in taz_data.columns if col.endswith('_pcnt')]
    for col in key_columns:
        if col in taz_data.columns:
            logging.info(f"{col} summary: \n{taz_data[col].describe()}")
        else:
            logging.warning(f"{col} not found in final data")

def main():
    taz_gdf, census_gdf = load_data()
    taz_gdf, census_gdf = prepare_data(taz_gdf, census_gdf)
    allocated_gdf = spatial_allocation(taz_gdf, census_gdf)
    taz_data = aggregate_to_taz(allocated_gdf)
    validate_results(taz_data, census_gdf)
    taz_data.to_csv(OUTPUT_CSV)
    logging.info(f"File saved successfully at: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()