"""
Import Existing Data from CSV Files to Database
Migrates data from data/fix/*.csv files into the web application database
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app import create_app
from app.models import db, Regency, DengueCase, ClimateData, NDVIData

def import_csv_to_database(csv_path, source_name='imported'):
    """
    Import data from CSV file to database
    
    Args:
        csv_path: Path to CSV file
        source_name: Data source identifier
    """
    print(f"\n{'='*60}")
    print(f"Importing data from: {csv_path}")
    print(f"{'='*60}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Get all regencies
    regencies = {r.name: r.id for r in Regency.query.all()}
    print(f"Found {len(regencies)} regencies in database")
    
    imported_dengue = 0
    imported_climate = 0
    imported_ndvi = 0
    updated_dengue = 0
    updated_climate = 0
    updated_ndvi = 0
    skipped = 0
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Get regency ID
            regency_name = row['Region']
            if regency_name not in regencies:
                print(f"Warning: Regency '{regency_name}' not found in database. Skipping row {idx+1}")
                skipped += 1
                continue
            
            regency_id = regencies[regency_name]
            year = int(row['Year'])
            month = int(row['Month'])
            
            # 1. Import Dengue Cases
            cases = int(row['Cases'])
            existing_case = DengueCase.query.filter_by(
                regency_id=regency_id,
                year=year,
                month=month
            ).first()
            
            if existing_case:
                existing_case.cases = cases
                existing_case.data_source = source_name
                updated_dengue += 1
            else:
                new_case = DengueCase(
                    regency_id=regency_id,
                    year=year,
                    month=month,
                    cases=cases,
                    data_source=source_name
                )
                db.session.add(new_case)
                imported_dengue += 1
            
            # 2. Import Climate Data
            existing_climate = ClimateData.query.filter_by(
                regency_id=regency_id,
                year=year,
                month=month
            ).first()
            
            climate_data = {
                'temperature_min': float(row['Temperature_Min']) if pd.notna(row['Temperature_Min']) else None,
                'temperature_max': float(row['Temperature_Max']) if pd.notna(row['Temperature_Max']) else None,
                'temperature_avg': float(row['Temperature_Avg']) if pd.notna(row['Temperature_Avg']) else None,
                'humidity': float(row['Humidity']) if pd.notna(row['Humidity']) else None,
                'precipitation_total': float(row['Precipitation_Total']) if pd.notna(row['Precipitation_Total']) else None,
                'pressure': float(row['Pressure']) if pd.notna(row['Pressure']) else None,
                'wind_speed': float(row['Wind_Speed']) if pd.notna(row['Wind_Speed']) else None,
                'wind_direction': float(row['Wind_Direction']) if pd.notna(row['Wind_Direction']) else None,
                'cloud_cover': float(row['Cloud_Cover']) if pd.notna(row['Cloud_Cover']) else None,
                'data_source': source_name
            }
            
            if existing_climate:
                for key, value in climate_data.items():
                    setattr(existing_climate, key, value)
                updated_climate += 1
            else:
                new_climate = ClimateData(
                    regency_id=regency_id,
                    year=year,
                    month=month,
                    **climate_data
                )
                db.session.add(new_climate)
                imported_climate += 1
            
            # 3. Import NDVI Data
            if 'NDVI' in row and pd.notna(row['NDVI']):
                existing_ndvi = NDVIData.query.filter_by(
                    regency_id=regency_id,
                    year=year,
                    month=month
                ).first()
                
                ndvi_value = float(row['NDVI'])
                
                if existing_ndvi:
                    existing_ndvi.ndvi_value = ndvi_value
                    existing_ndvi.data_source = source_name
                    existing_ndvi.is_imputed = False
                    updated_ndvi += 1
                else:
                    new_ndvi = NDVIData(
                        regency_id=regency_id,
                        year=year,
                        month=month,
                        ndvi_value=ndvi_value,
                        data_source=source_name,
                        is_imputed=False
                    )
                    db.session.add(new_ndvi)
                    imported_ndvi += 1
            
            # Commit every 50 rows
            if (idx + 1) % 50 == 0:
                db.session.commit()
                print(f"Processed {idx+1} rows...")
        
        except Exception as e:
            print(f"Error processing row {idx+1}: {str(e)}")
            db.session.rollback()
            skipped += 1
            continue
    
    # Final commit
    db.session.commit()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Import Summary:")
    print(f"{'='*60}")
    print(f"Dengue Cases:  {imported_dengue} imported, {updated_dengue} updated")
    print(f"Climate Data:  {imported_climate} imported, {updated_climate} updated")
    print(f"NDVI Data:     {imported_ndvi} imported, {updated_ndvi} updated")
    print(f"Skipped:       {skipped} rows")
    print(f"{'='*60}\n")


def main():
    """Main import function"""
    print("\n" + "="*60)
    print("DATA IMPORT UTILITY")
    print("Importing existing CSV data to database")
    print("="*60 + "\n")
    
    # Create app context
    app = create_app('development')
    
    with app.app_context():
        # Check if regencies exist
        regency_count = Regency.query.count()
        if regency_count == 0:
            print("ERROR: No regencies found in database!")
            print("Please run the application first to initialize default data.")
            return
        
        print(f"Found {regency_count} regencies in database")
        
        # Define CSV files to import (relative to project root)
        csv_files = [
            {
                'path': '../data/fix/data_monthly_5kab_2021_2024_ndvi.csv',
                'source': 'historical_2021_2024'
            },
            {
                'path': '../data/fix/data_monthly_5kab_2025_ndvi.csv',
                'source': 'current_2025'
            }
        ]
        
        total_imported = 0
        
        for csv_info in csv_files:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                csv_info['path']
            )
            
            if os.path.exists(csv_path):
                import_csv_to_database(csv_path, csv_info['source'])
                total_imported += 1
            else:
                print(f"WARNING: File not found: {csv_path}")
        
        print(f"\n{'='*60}")
        print(f"IMPORT COMPLETE!")
        print(f"Imported data from {total_imported} CSV files")
        print(f"{'='*60}\n")
        
        # Print database statistics
        print("Database Statistics:")
        print(f"  Dengue Cases: {DengueCase.query.count()}")
        print(f"  Climate Data: {ClimateData.query.count()}")
        print(f"  NDVI Data:    {NDVIData.query.count()}")
        print()


if __name__ == '__main__':
    main()
