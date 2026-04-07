import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_coordinates_from_csv(csv_path):
    """
    Load coordinates and location information from CSV file
    
    Parameters:
    csv_path (str): Path to CSV file containing location data
    
    Returns:
    pandas.DataFrame: DataFrame with location information
    list: List of (longitude, latitude) tuples
    """
    try:
        # Read CSV file
        locations_df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_columns = ['Kabupaten', 'Latitude', 'Longitude']
        missing_columns = [col for col in required_columns if col not in locations_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert coordinates to list of tuples
        coordinates = list(zip(locations_df['Longitude'], locations_df['Latitude']))
        
        return locations_df, coordinates
    except Exception as e:
        raise Exception(f"Error loading coordinates from CSV: {e}")

def load_color_table(act_file):
    """
    Load MODIS NDVI color table from binary .act file
    """
    try:
        # Read the binary file
        with open(act_file, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)
        
        # Reshape to RGB format (256 entries x 3 colors)
        colors = data.reshape(-1, 3)
        
        # Normalize to 0-1 range
        colors = colors.astype(np.float32) / 255.0
        
        # Create and return the colormap
        return LinearSegmentedColormap.from_list('modis_ndvi', colors)
    except Exception as e:
        print(f"Error loading color table: {e}")
        # Return a default colormap if loading fails
        return plt.cm.RdYlGn

def extract_ndvi_values(tiff_file, coordinates):
    """
    Extract NDVI values from specific coordinates in a TIFF file
    
    Parameters:
    tiff_file (str): Path to TIFF file
    coordinates (list): List of tuples containing (longitude, latitude)
    
    Returns:
    dict: Dictionary with coordinates as keys and NDVI values as values
    """
    try:
        with rasterio.open(tiff_file) as src:
            values = {}
            for coord in coordinates:
                # Convert geographic coordinates to pixel coordinates
                row, col = rowcol(src.transform, coord[0], coord[1])
                
                # Check if coordinates are within bounds
                if 0 <= row < src.height and 0 <= col < src.width:
                    value = src.read(1)[row, col]
                    # Convert to NDVI value (assuming standard scaling)
                    ndvi = value * 0.0001 if value != src.nodata else np.nan
                    values[coord] = ndvi
                else:
                    values[coord] = np.nan
                    print(f"Warning: Coordinates {coord} are outside the image bounds")
            
            return values
    except Exception as e:
        print(f"Error processing {tiff_file}: {e}")
        return {coord: np.nan for coord in coordinates}

def process_tiff_files(tiff_directory, locations_df, coordinates, color_table_path):
    """
    Process multiple TIFF files and extract NDVI values
    
    Parameters:
    tiff_directory (str): Directory containing TIFF files
    locations_df (pandas.DataFrame): DataFrame containing location information
    coordinates (list): List of tuples containing (longitude, latitude)
    color_table_path (str): Path to MODIS NDVI color table file
    
    Returns:
    pandas.DataFrame: DataFrame containing NDVI values for each location and date
    """
    # Load color table
    color_map = load_color_table(color_table_path)
    
    # Initialize results list
    results = []
    
    # Process each TIFF file
    tiff_files = [f for f in sorted(os.listdir(tiff_directory)) 
                 if f.lower().endswith(('.tif', '.tiff'))]
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {tiff_directory}")
    
    for filename in tiff_files:
        file_path = os.path.join(tiff_directory, filename)
        print(f"Processing {filename}...")
        
        try:
            # Extract date from filename format "MOD_NDVI_16_2023-11-17"
            date_str = filename.split('_')[3].split('.')[0]  # Get "2023-11-17"
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except Exception as e:
            print(f"Warning: Could not parse date from {filename}: {e}")
            date = None
        
        # Extract NDVI values
        values = extract_ndvi_values(file_path, coordinates)
        
        # Add to results - one row per location per date
        for coord, ndvi_value in values.items():
            # Find matching location information
            location_info = locations_df[
                (locations_df['Longitude'] == coord[0]) & 
                (locations_df['Latitude'] == coord[1])
            ].iloc[0]
            
            row_data = {
                'Date': date,
                'Kabupaten': location_info['Kabupaten'],
                'Longitude': coord[0],
                'Latitude': coord[1],
                'NDVI': ndvi_value
            }
            results.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df, color_map

def visualize_ndvi_timeseries(df, color_map, output_path):
    """
    Create visualization of NDVI time series
    """
    try:
        plt.figure(figsize=(15, 8))
        
        # Create scatter plot with NDVI values
        scatter = plt.scatter([], [], c=[], cmap=color_map, vmin=-1, vmax=1)
        
        # Plot time series for each unique location
        for (kab), group in df.groupby(['Kabupaten']):
            label = f'{kab}'
            plt.plot(group['Date'], group['NDVI'], 
                    label=label, marker='o', linestyle='-')
        
        # Add colorbar using the scatter object
        plt.colorbar(scatter, label='NDVI Value')
        
        plt.xlabel('Date')
        plt.ylabel('NDVI')
        plt.title('NDVI Time Series by Location')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    # Define your parameters
    tiff_directory = 'MOD_NDVI_16_2025_2026'  # Replace with your TIFF directory path
    color_table_path = 'modis_ndvi.act'  # Replace with your .act file path
    locations_csv = 'list_kabupaten.csv'  # Replace with your CSV file path
    
    output_csv = 'ndvi_results_kabupaten_2025_2026.csv'
    output_plot = 'ndvi_timeseries_kabupaten_2025_2026.png'
    
    try:
        # Load coordinates from CSV
        print("Loading location data...")
        locations_df, coordinates = load_coordinates_from_csv(locations_csv)
        
        # Process files
        print("Starting NDVI extraction...")
        df, color_map = process_tiff_files(tiff_directory, locations_df, coordinates, color_table_path)
        
        # Sort the DataFrame
        df = df.sort_values(['Date', 'Kabupaten'])
        
        # Save results
        df.to_csv(output_csv, index=False)
        print(f'Results saved to {output_csv}')
        
        # Create visualization
        visualize_ndvi_timeseries(df, color_map, output_plot)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()