"""
Unified Data Pipeline Service
Simplifies the complex multi-step data processing into a single integrated service
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
from epiweeks import Week
import warnings
warnings.filterwarnings('ignore')

from ..models import db, Regency, DengueCase, ClimateData, NDVIData, DataProcessingLog


class DataPipelineService:
    """
    Unified service to manage all data processing:
    1. Manual dengue case input
    2. Automated climate data fetching
    3. Automated NDVI processing
    4. Automatic aggregation to monthly
    """
    
    def __init__(self, config):
        self.config = config
        self.openweather_api_key = config.OPENWEATHER_API_KEY
        self.nasa_power_url = config.NASA_POWER_API_URL
        self.api_delay = config.API_DELAY_SECONDS
        
    # ==========================================
    # 1. DENGUE CASE MANAGEMENT
    # ==========================================
    
    def add_dengue_cases(self, regency_id: int, year: int, month: int, 
                        cases: int, user_id: int, notes: str = None) -> Dict:
        """
        Add or update dengue cases for a specific regency and month
        Replaces the need to download from SKDR
        
        Args:
            regency_id: ID of the regency
            year: Year (e.g., 2024)
            month: Month (1-12)
            cases: Number of dengue cases
            user_id: ID of user entering the data
            notes: Optional notes
            
        Returns:
            Dictionary with status and message
        """
        try:
            # Check if entry exists
            existing = DengueCase.query.filter_by(
                regency_id=regency_id,
                year=year,
                month=month
            ).first()
            
            if existing:
                # Update existing entry
                existing.cases = cases
                existing.notes = notes
                existing.updated_at = datetime.utcnow()
                existing.reported_by_id = user_id
                action = 'updated'
            else:
                # Create new entry
                new_case = DengueCase(
                    regency_id=regency_id,
                    year=year,
                    month=month,
                    cases=cases,
                    data_source='manual',
                    reported_by_id=user_id,
                    notes=notes
                )
                db.session.add(new_case)
                action = 'added'
            
            db.session.commit()
            
            return {
                'success': True,
                'action': action,
                'message': f'Dengue cases {action} successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def bulk_import_dengue_cases(self, csv_file_path: str, user_id: int) -> Dict:
        """
        Bulk import dengue cases from CSV file
        
        CSV format: Year,Region,Month,Cases
        
        Args:
            csv_file_path: Path to CSV file
            user_id: ID of user importing data
            
        Returns:
            Dictionary with import statistics
        """
        try:
            df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_cols = ['Year', 'Region', 'Month', 'Cases']
            if not all(col in df.columns for col in required_cols):
                return {
                    'success': False,
                    'message': f'CSV must contain columns: {", ".join(required_cols)}'
                }
            
            imported = 0
            updated = 0
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    # Find regency by name
                    regency = Regency.query.filter_by(name=row['Region']).first()
                    if not regency:
                        errors.append(f"Row {idx+1}: Regency '{row['Region']}' not found")
                        continue
                    
                    result = self.add_dengue_cases(
                        regency_id=regency.id,
                        year=int(row['Year']),
                        month=int(row['Month']),
                        cases=int(row['Cases']),
                        user_id=user_id
                    )
                    
                    if result['success']:
                        if result['action'] == 'added':
                            imported += 1
                        else:
                            updated += 1
                    else:
                        errors.append(f"Row {idx+1}: {result['message']}")
                        
                except Exception as e:
                    errors.append(f"Row {idx+1}: {str(e)}")
            
            return {
                'success': True,
                'imported': imported,
                'updated': updated,
                'errors': errors,
                'total_rows': len(df)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error reading CSV: {str(e)}'
            }
    
    # ==========================================
    # 2. CLIMATE DATA FETCHING
    # ==========================================
    
    def fetch_climate_data_for_month(self, regency_id: int, year: int, month: int) -> Dict:
        """
        Fetch climate data from APIs for a specific regency and month
        Integrates functionality from get_climate_data_v3.py
        
        Args:
            regency_id: ID of the regency
            year: Year
            month: Month
            
        Returns:
            Dictionary with climate data or error
        """
        try:
            # Get regency info
            regency = Regency.query.get(regency_id)
            if not regency:
                return {'success': False, 'message': 'Regency not found'}
            
            # Get date range for the month
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
            # Try NASA POWER API first
            climate_data = self._fetch_nasa_power(
                regency.latitude,
                regency.longitude,
                start_date,
                end_date
            )
            
            if not climate_data:
                # Fallback to OpenWeather if NASA fails
                climate_data = self._fetch_openweather(
                    regency.latitude,
                    regency.longitude,
                    start_date,
                    end_date
                )
            
            if not climate_data:
                return {'success': False, 'message': 'Failed to fetch climate data from all sources'}
            
            # Calculate monthly averages
            monthly_avg = {
                'temperature_min': np.mean(climate_data.get('temp_min', [])),
                'temperature_max': np.mean(climate_data.get('temp_max', [])),
                'temperature_avg': np.mean(climate_data.get('temp_avg', [])),
                'humidity': np.mean(climate_data.get('humidity', [])),
                'precipitation_total': np.sum(climate_data.get('precipitation', [])),
                'pressure': np.mean(climate_data.get('pressure', [])),
                'wind_speed': np.mean(climate_data.get('wind_speed', [])),
                'wind_direction': np.mean(climate_data.get('wind_direction', [])),
                'cloud_cover': np.mean(climate_data.get('cloud_cover', []))
            }
            
            # Save to database
            existing = ClimateData.query.filter_by(
                regency_id=regency_id,
                year=year,
                month=month
            ).first()
            
            if existing:
                for key, value in monthly_avg.items():
                    setattr(existing, key, value)
                existing.data_source = climate_data.get('source', 'unknown')
                existing.fetched_at = datetime.utcnow()
            else:
                new_climate = ClimateData(
                    regency_id=regency_id,
                    year=year,
                    month=month,
                    **monthly_avg,
                    data_source=climate_data.get('source', 'unknown')
                )
                db.session.add(new_climate)
            
            db.session.commit()
            
            return {
                'success': True,
                'data': monthly_avg,
                'source': climate_data.get('source', 'unknown')
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def _fetch_nasa_power(self, lat: float, lon: float, start_date: datetime, 
                         end_date: datetime) -> Optional[Dict]:
        """Fetch data from NASA POWER API"""
        try:
            params = {
                'parameters': 'T2M,T2M_MIN,T2M_MAX,RH2M,PRECTOTCORR,PS,WS2M',
                'community': 'AG',
                'longitude': lon,
                'latitude': lat,
                'start': start_date.strftime('%Y%m%d'),
                'end': end_date.strftime('%Y%m%d'),
                'format': 'JSON'
            }
            
            response = requests.get(self.nasa_power_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                parameters = data.get('properties', {}).get('parameter', {})
                
                return {
                    'temp_min': list(parameters.get('T2M_MIN', {}).values()),
                    'temp_max': list(parameters.get('T2M_MAX', {}).values()),
                    'temp_avg': list(parameters.get('T2M', {}).values()),
                    'humidity': list(parameters.get('RH2M', {}).values()),
                    'precipitation': list(parameters.get('PRECTOTCORR', {}).values()),
                    'pressure': list(parameters.get('PS', {}).values()),
                    'wind_speed': list(parameters.get('WS2M', {}).values()),
                    'wind_direction': [0] * len(list(parameters.get('WS2M', {}).values())),  # NASA doesn't provide direction
                    'cloud_cover': [50] * len(list(parameters.get('WS2M', {}).values())),  # Default estimate
                    'source': 'nasa_power'
                }
            
            return None
            
        except Exception as e:
            print(f"NASA POWER API error: {str(e)}")
            return None
    
    def _fetch_openweather(self, lat: float, lon: float, start_date: datetime, 
                          end_date: datetime) -> Optional[Dict]:
        """Fetch data from OpenWeather API"""
        try:
            # OpenWeather One Call API 3.0 Day Summary
            # Note: This requires a subscription for historical data
            # Simplified implementation - you may need to adjust based on your API plan
            
            url = 'https://api.openweathermap.org/data/3.0/onecall/day_summary'
            
            # Collect data for each day
            all_data = {
                'temp_min': [],
                'temp_max': [],
                'temp_avg': [],
                'humidity': [],
                'precipitation': [],
                'pressure': [],
                'wind_speed': [],
                'wind_direction': [],
                'cloud_cover': []
            }
            
            current_date = start_date
            while current_date <= end_date:
                params = {
                    'lat': lat,
                    'lon': lon,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'appid': self.openweather_api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    all_data['temp_min'].append(data.get('temperature', {}).get('min', 25))
                    all_data['temp_max'].append(data.get('temperature', {}).get('max', 30))
                    all_data['temp_avg'].append(data.get('temperature', {}).get('afternoon', 28))
                    all_data['humidity'].append(data.get('humidity', {}).get('afternoon', 70))
                    all_data['precipitation'].append(data.get('precipitation', {}).get('total', 0))
                    all_data['pressure'].append(data.get('pressure', {}).get('afternoon', 1010))
                    all_data['wind_speed'].append(data.get('wind', {}).get('max', {}).get('speed', 5))
                    all_data['wind_direction'].append(data.get('wind', {}).get('max', {}).get('direction', 180))
                    all_data['cloud_cover'].append(data.get('cloud_cover', {}).get('afternoon', 50))
                
                current_date += timedelta(days=1)
                
                # Rate limiting
                import time
                time.sleep(self.api_delay)
            
            all_data['source'] = 'openweather'
            return all_data
            
        except Exception as e:
            print(f"OpenWeather API error: {str(e)}")
            return None
    
    def fetch_all_climate_data(self, year: int, month: int, user_id: int) -> Dict:
        """
        Fetch climate data for all regencies for a specific month
        One-click operation for admin
        
        Args:
            year: Year
            month: Month
            user_id: User performing the operation
            
        Returns:
            Dictionary with processing results
        """
        # Create processing log
        log = DataProcessingLog(
            user_id=user_id,
            process_type='climate_fetch',
            status='started',
            details={'year': year, 'month': month}
        )
        db.session.add(log)
        db.session.commit()
        
        results = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            regencies = Regency.query.filter_by(is_active=True).all()
            
            for regency in regencies:
                result = self.fetch_climate_data_for_month(regency.id, year, month)
                
                if result['success']:
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{regency.name}: {result['message']}")
                
                # Rate limiting
                import time
                time.sleep(self.api_delay)
            
            log.status = 'completed'
            log.records_processed = results['success']
            log.completed_at = datetime.utcnow()
            db.session.commit()
            
            return results
            
        except Exception as e:
            log.status = 'failed'
            log.error_message = str(e)
            log.completed_at = datetime.utcnow()
            db.session.commit()
            
            return {
                'success': 0,
                'failed': len(Regency.query.filter_by(is_active=True).all()),
                'errors': [str(e)]
            }
    
    # ==========================================
    # 3. NDVI DATA PROCESSING
    # ==========================================
    
    def process_ndvi_from_satellite(self, tiff_file_path: str, year: int, 
                                    month: int, user_id: int) -> Dict:
        """
        Process NDVI data from uploaded GeoTIFF file
        Simplified version of NEO/get_ndvi_kabupaten.py
        
        Args:
            tiff_file_path: Path to GeoTIFF file
            year: Year
            month: Month
            user_id: User performing the operation
            
        Returns:
            Dictionary with processing results
        """
        try:
            import rasterio
            from rasterio.windows import from_bounds
            
            results = {
                'success': 0,
                'failed': 0,
                'errors': []
            }
            
            # Open GeoTIFF
            with rasterio.open(tiff_file_path) as src:
                regencies = Regency.query.filter_by(is_active=True).all()
                
                for regency in regencies:
                    try:
                        # Get pixel value at regency location
                        row, col = src.index(regency.longitude, regency.latitude)
                        ndvi_value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                        
                        # Validate NDVI value (-1 to 1 range)
                        if -1 <= ndvi_value <= 1:
                            # Save to database
                            existing = NDVIData.query.filter_by(
                                regency_id=regency.id,
                                year=year,
                                month=month
                            ).first()
                            
                            if existing:
                                existing.ndvi_value = float(ndvi_value)
                                existing.processing_date = datetime.utcnow()
                                existing.is_imputed = False
                            else:
                                new_ndvi = NDVIData(
                                    regency_id=regency.id,
                                    year=year,
                                    month=month,
                                    ndvi_value=float(ndvi_value),
                                    data_source='modis',
                                    is_imputed=False
                                )
                                db.session.add(new_ndvi)
                            
                            results['success'] += 1
                        else:
                            results['failed'] += 1
                            results['errors'].append(f"{regency.name}: Invalid NDVI value {ndvi_value}")
                    
                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(f"{regency.name}: {str(e)}")
            
            db.session.commit()
            
            # Create processing log
            log = DataProcessingLog(
                user_id=user_id,
                process_type='ndvi_process',
                status='completed',
                records_processed=results['success'],
                details={'year': year, 'month': month},
                completed_at=datetime.utcnow()
            )
            db.session.add(log)
            db.session.commit()
            
            return results
            
        except Exception as e:
            return {
                'success': 0,
                'failed': 0,
                'errors': [f'Error processing NDVI file: {str(e)}']
            }
    
    def impute_missing_ndvi(self, year: int, month: int) -> Dict:
        """
        Fill missing NDVI values using forward/backward fill
        Simplified version of NEO/ndvi_imputation.py
        
        Args:
            year: Year to impute
            month: Month to impute
            
        Returns:
            Dictionary with imputation results
        """
        try:
            regencies = Regency.query.filter_by(is_active=True).all()
            imputed_count = 0
            
            for regency in regencies:
                # Check if NDVI data exists for this month
                existing = NDVIData.query.filter_by(
                    regency_id=regency.id,
                    year=year,
                    month=month
                ).first()
                
                if not existing or existing.ndvi_value is None:
                    # Try forward fill: get most recent NDVI value
                    recent = NDVIData.query.filter(
                        NDVIData.regency_id == regency.id,
                        NDVIData.year * 12 + NDVIData.month < year * 12 + month,
                        NDVIData.ndvi_value.isnot(None)
                    ).order_by(
                        NDVIData.year.desc(),
                        NDVIData.month.desc()
                    ).first()
                    
                    if recent:
                        if existing:
                            existing.ndvi_value = recent.ndvi_value
                            existing.is_imputed = True
                        else:
                            new_ndvi = NDVIData(
                                regency_id=regency.id,
                                year=year,
                                month=month,
                                ndvi_value=recent.ndvi_value,
                                data_source='imputed',
                                is_imputed=True
                            )
                            db.session.add(new_ndvi)
                        
                        imputed_count += 1
            
            db.session.commit()
            
            return {
                'success': True,
                'imputed': imputed_count,
                'message': f'Imputed {imputed_count} missing NDVI values'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': f'Error during imputation: {str(e)}'
            }
    
    # ==========================================
    # 4. DATA EXPORT
    # ==========================================
    
    def export_to_csv(self, year_start: int, year_end: int, output_path: str) -> Dict:
        """
        Export all data to CSV format matching the original structure
        Format: Year,Region,Month,Cases,Latitude,Longitude,NDVI,Climate_Variables...
        
        Args:
            year_start: Start year
            year_end: End year
            output_path: Path to save CSV file
            
        Returns:
            Dictionary with export results
        """
        try:
            records = []
            
            regencies = Regency.query.filter_by(is_active=True).all()
            
            for year in range(year_start, year_end + 1):
                for month in range(1, 13):
                    for regency in regencies:
                        # Get dengue cases
                        dengue = DengueCase.query.filter_by(
                            regency_id=regency.id,
                            year=year,
                            month=month
                        ).first()
                        
                        # Get climate data
                        climate = ClimateData.query.filter_by(
                            regency_id=regency.id,
                            year=year,
                            month=month
                        ).first()
                        
                        # Get NDVI data
                        ndvi = NDVIData.query.filter_by(
                            regency_id=regency.id,
                            year=year,
                            month=month
                        ).first()
                        
                        # Create record
                        record = {
                            'Year': year,
                            'Region': regency.name,
                            'Month': month,
                            'Cases': dengue.cases if dengue else 0,
                            'Latitude': regency.latitude,
                            'Longitude': regency.longitude,
                            'NDVI': ndvi.ndvi_value if ndvi else None,
                            'Cloud_Cover': climate.cloud_cover if climate else None,
                            'Humidity': climate.humidity if climate else None,
                            'Precipitation_Total': climate.precipitation_total if climate else None,
                            'Temperature_Min': climate.temperature_min if climate else None,
                            'Temperature_Max': climate.temperature_max if climate else None,
                            'Temperature_Avg': climate.temperature_avg if climate else None,
                            'Pressure': climate.pressure if climate else None,
                            'Wind_Speed': climate.wind_speed if climate else None,
                            'Wind_Direction': climate.wind_direction if climate else None
                        }
                        
                        records.append(record)
            
            # Create DataFrame and save
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            
            return {
                'success': True,
                'records': len(records),
                'file_path': output_path,
                'message': f'Exported {len(records)} records to {output_path}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error exporting data: {str(e)}'
            }
