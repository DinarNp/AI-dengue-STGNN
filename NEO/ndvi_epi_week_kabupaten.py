import pandas as pd
import numpy as np
from datetime import datetime, date
from epiweeks import Week
import os

def load_ndvi_data(file_path):
    """
    Load NDVI data from CSV file and convert date to datetime
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return None
            
        print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Successfully read CSV with {len(df)} rows")
        
        df['Date'] = pd.to_datetime(df['Date'])
        print("Date conversion successful")
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_epi_week_info(date):
    """
    Get epidemiological week information using epiweeks package
    Returns tuple of (epi_year, epi_week)
    """
    epi_week = Week.fromdate(date)
    return epi_week.year, epi_week.week

def get_week_dates(year, week):
    """
    Get start and end dates for a week, handling potential invalid weeks safely
    """
    try:
        week_obj = Week(year, week)
        return pd.Timestamp(week_obj.startdate()), pd.Timestamp(week_obj.enddate())
    except ValueError as e:
        print(f"Warning: Invalid week {week} for year {year}: {str(e)}")
        return pd.NaT, pd.NaT

def create_complete_calendar(start_year, end_year, locations):
    """
    Create a complete calendar with weeks 1-52 for all locations
    """
    print(f"Creating calendar for years {start_year}-{end_year} with {len(locations)} locations")
    calendar_rows = []
    
    for year in range(start_year, end_year + 1):
        for week_num in range(1, 53):
            week_start, week_end = get_week_dates(year, week_num)
            
            for location in locations:
                calendar_rows.append({
                    'EpiYear': year,
                    'EpiWeek': week_num,
                    'YearWeek': f"{year}-W{week_num:02d}",
                    'WeekStart': week_start,
                    'WeekEnd': week_end,
                    'Kabupaten': location['Kabupaten'],
                    'Longitude': location['Longitude'],
                    'Latitude': location['Latitude'],
                    'NDVI': None
                })
    
    print(f"Created calendar with {len(calendar_rows)} total entries")
    df = pd.DataFrame(calendar_rows)
    
    # Convert date columns to datetime
    df['WeekStart'] = pd.to_datetime(df['WeekStart'])
    df['WeekEnd'] = pd.to_datetime(df['WeekEnd'])
    
    return df

def process_ndvi_data(df):
    """
    Process NDVI data and add epidemiological week information
    """
    print("Processing NDVI data...")
    epi_info = df['Date'].apply(get_epi_week_info)
    df['EpiYear'] = epi_info.apply(lambda x: x[0])
    df['EpiWeek'] = epi_info.apply(lambda x: x[1])
    df['YearWeek'] = df.apply(lambda x: f"{x['EpiYear']}-W{x['EpiWeek']:02d}", axis=1)
    print("NDVI data processing complete")
    return df

def merge_data_with_calendar(calendar_df, data_df):
    """
    Merge the complete calendar with available NDVI data
    """
    print("Merging calendar with NDVI data...")
    merge_key = ['Kabupaten', 'EpiYear', 'EpiWeek']
    
    merged_df = calendar_df.merge(
        data_df[merge_key + ['NDVI']],
        on=merge_key,
        how='left',
        suffixes=('_calendar', '')
    )
    
    print(f"Merge complete. Result has {len(merged_df)} rows")
    return merged_df

def main():
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    input_file = 'ndvi_results_kabupaten_2025_2026.csv'
    output_file = 'ndvi_complete_calendar_kabupaten_2025_2026.csv'
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    try:
        # Load data
        print("\nStep 1: Loading NDVI data...")
        df = load_ndvi_data(input_file)
        
        if df is not None:
            # Process NDVI data
            print("\nStep 2: Processing NDVI data...")
            df = process_ndvi_data(df)
            
            # Get unique locations with their coordinates
            locations = df[['Kabupaten', 'Longitude', 'Latitude']].drop_duplicates().to_dict('records')
            print(f"Found {len(locations)} unique locations")
            
            # Get year range
            start_year = 2025
            end_year = 2025
            print(f"Year range: {start_year}-{end_year}")
            
            # Create complete calendar
            print("\nStep 3: Creating complete calendar...")
            calendar_df = create_complete_calendar(start_year, end_year, locations)
            
            # Merge data with calendar
            print("\nStep 4: Merging data with calendar...")
            result_df = merge_data_with_calendar(calendar_df, df)
            
            # Sort the results
            print("\nStep 5: Sorting results...")
            result_df = result_df.sort_values(['EpiYear', 'Kabupaten', 'EpiWeek'])
            
            # Save results
            print(f"\nStep 6: Saving results to {output_file}...")
            result_df.to_csv(output_file, index=False)
            print(f"Results saved successfully to {output_file}")
            
            # Print summary
            print("\nSummary:")
            total_weeks = len(result_df)
            data_weeks = result_df['NDVI'].notna().sum()
            print(f"Total weeks in calendar: {total_weeks}")
            print(f"Weeks with data: {data_weeks}")
            print(f"Weeks without data: {total_weeks - data_weeks}")
            print(f"Data coverage: {(data_weeks/total_weeks)*100:.1f}%")
            
            # Print week range
            print("\nDate range:")
            print(f"Start: {result_df['WeekStart'].min()}")
            print(f"End: {result_df['WeekEnd'].max()}")
            
        else:
            print("Failed to load data.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("Full error trace:")
        print(traceback.format_exc())

if __name__ == '__main__':
    main()