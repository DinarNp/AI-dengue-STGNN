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
        self.openweather_api_key = config.get('OPENWEATHER_API_KEY')
        self.api_delay = config.get('API_DELAY_SECONDS', 1.5)
        
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
    # 2. CLIMATE DATA FETCHING (OpenWeatherMap only)
    # ==========================================

    def fetch_climate_data_for_month(self, regency_id: int, year: int, month: int) -> Dict:
        """
        Fetch climate data for a specific regency and month using OpenWeatherMap
        One Call API 3.0 day_summary endpoint (one call per day, then average).
        Monthly aggregation matches convert_weekly_to_monthly.py: mean for all vars.
        """
        try:
            regency = Regency.query.get(regency_id)
            if not regency:
                return {'success': False, 'message': 'Regency not found'}

            if not self.openweather_api_key:
                return {'success': False, 'message': 'OpenWeatherMap API key not configured'}

            start_date = datetime(year, month, 1)
            end_date   = datetime(year + 1, 1, 1) - timedelta(days=1) if month == 12 \
                         else datetime(year, month + 1, 1) - timedelta(days=1)

            monthly_avg = self._fetch_openweather_month(
                regency.latitude, regency.longitude, start_date, end_date
            )

            if not any(v is not None for v in monthly_avg.values()):
                return {'success': False, 'message': 'OpenWeatherMap returned no data for this period'}

            db_fields = {
                'temperature_min':     monthly_avg.get('Temperature_Min'),
                'temperature_max':     monthly_avg.get('Temperature_Max'),
                'temperature_avg':     monthly_avg.get('Temperature_Avg'),
                'humidity':            monthly_avg.get('Humidity'),
                'precipitation_total': monthly_avg.get('Precipitation_Total'),
                'pressure':            monthly_avg.get('Pressure'),
                'wind_speed':          monthly_avg.get('Wind_Speed'),
                'wind_direction':      monthly_avg.get('Wind_Direction'),
                'cloud_cover':         monthly_avg.get('Cloud_Cover'),
            }

            existing = ClimateData.query.filter_by(
                regency_id=regency_id, year=year, month=month
            ).first()

            if existing:
                for col, val in db_fields.items():
                    if val is not None:
                        setattr(existing, col, val)
                existing.data_source = 'openweather'
                existing.fetched_at  = datetime.utcnow()
            else:
                db.session.add(ClimateData(
                    regency_id=regency_id, year=year, month=month,
                    data_source='openweather',
                    **{k: v for k, v in db_fields.items() if v is not None}
                ))

            db.session.commit()
            return {'success': True, 'data': db_fields, 'source': 'openweather'}

        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': f'Error: {str(e)}'}

    def _fetch_openweather_month(self, lat: float, lon: float,
                                 start_date: datetime, end_date: datetime) -> Dict:
        """
        Call OpenWeatherMap One Call API 3.0 day_summary once per day for the
        given date range, then average all daily values for the month.

        Endpoint: GET https://api.openweathermap.org/data/3.0/onecall/day_summary
        Params  : lat, lon, date (YYYY-MM-DD), appid, units=metric

        Response fields used:
            temperature.min/max/afternoon → Temperature_Min/Max/Avg
            humidity.afternoon            → Humidity
            precipitation.total           → Precipitation_Total
            pressure.afternoon            → Pressure
            wind.max.speed/direction      → Wind_Speed/Direction
            cloud_cover.afternoon         → Cloud_Cover

        Aggregation matches convert_weekly_to_monthly.py: mean for all climate vars.
        """
        import time

        url   = 'https://api.openweathermap.org/data/3.0/onecall/day_summary'
        accum = {k: [] for k in [
            'Temperature_Min', 'Temperature_Max', 'Temperature_Avg',
            'Humidity', 'Precipitation_Total', 'Pressure',
            'Wind_Speed', 'Wind_Direction', 'Cloud_Cover'
        ]}

        current = start_date
        while current <= end_date:
            try:
                resp = requests.get(url, params={
                    'lat':   lat,
                    'lon':   lon,
                    'date':  current.strftime('%Y-%m-%d'),
                    'appid': self.openweather_api_key,
                    'units': 'metric'   # Celsius, m/s, hPa
                }, timeout=30)

                if resp.status_code == 200:
                    d = resp.json()

                    temp   = d.get('temperature', {})
                    hum    = d.get('humidity', {})
                    pres   = d.get('pressure', {})
                    wind   = d.get('wind', {}).get('max', {})
                    cc     = d.get('cloud_cover', {})
                    precip = d.get('precipitation', {})

                    def _add(key, val):
                        if val is not None:
                            accum[key].append(val)

                    _add('Temperature_Min',     temp.get('min'))
                    _add('Temperature_Max',     temp.get('max'))
                    _add('Temperature_Avg',     temp.get('afternoon'))
                    _add('Humidity',            hum.get('afternoon'))
                    _add('Pressure',            pres.get('afternoon'))
                    _add('Wind_Speed',          wind.get('speed'))
                    _add('Wind_Direction',      wind.get('direction'))
                    _add('Cloud_Cover',         cc.get('afternoon'))
                    _add('Precipitation_Total', precip.get('total'))

                else:
                    print(f"OpenWeatherMap {resp.status_code} for {current.strftime('%Y-%m-%d')}: {resp.text[:200]}")

            except Exception as e:
                print(f"OpenWeatherMap error for {current.strftime('%Y-%m-%d')}: {e}")

            current += timedelta(days=1)
            time.sleep(self.api_delay)

        return {k: round(float(np.mean(v)), 4) if v else None for k, v in accum.items()}
    
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
    
    def _extract_ndvi_from_tiff(self, tiff_path: str, regencies) -> Dict:
        """
        Extract raw NDVI values for all regencies from a single GeoTIFF.
        Uses same logic as NEO/get_ndvi_kabupaten.py:
          - pixel value * 0.0001 → NDVI  (MODIS standard scaling)
          - nodata pixels → None
        Returns dict: {regency.id: float or None}
        """
        import rasterio
        values = {}
        with rasterio.open(tiff_path) as src:
            band = src.read(1)
            nodata = src.nodata
            for regency in regencies:
                try:
                    row, col = src.index(regency.longitude, regency.latitude)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        raw = band[row, col]
                        if nodata is not None and raw == nodata:
                            values[regency.id] = None
                        else:
                            ndvi = float(raw) * 0.0001
                            values[regency.id] = ndvi if -1.0 <= ndvi <= 1.0 else None
                    else:
                        values[regency.id] = None
                except Exception:
                    values[regency.id] = None
        return values

    def process_ndvi_two_tiffs(self, tiff1_path: str, tiff2_path: str,
                               year: int, month: int, user_id: int) -> Dict:
        """
        Process monthly NDVI from two GeoTIFF files (two 16-day composites per month).
        Mirrors NEO/get_ndvi_kabupaten.py extraction + monthly average logic.

        Steps:
          1. Extract NDVI per regency from file 1
          2. Extract NDVI per regency from file 2
          3. Average the two values → monthly NDVI
          4. Save/update NDVIData in DB

        Args:
            tiff1_path: Path to first GeoTIFF (first 16-day period)
            tiff2_path: Path to second GeoTIFF (second 16-day period)
            year: Target year
            month: Target month
            user_id: ID of user performing the upload

        Returns:
            Dictionary with processing results
        """
        try:
            import rasterio  # noqa: confirm available
        except ImportError:
            return {
                'success': 0,
                'failed': 0,
                'errors': ['rasterio is not installed. Run: pip install rasterio']
            }

        try:
            regencies = Regency.query.filter_by(is_active=True).all()

            values1 = self._extract_ndvi_from_tiff(tiff1_path, regencies)
            values2 = self._extract_ndvi_from_tiff(tiff2_path, regencies)

            results = {'success': 0, 'failed': 0, 'errors': []}

            for regency in regencies:
                v1 = values1.get(regency.id)
                v2 = values2.get(regency.id)

                # Average: use available values; skip if both missing
                available = [v for v in (v1, v2) if v is not None]
                if not available:
                    results['failed'] += 1
                    results['errors'].append(f"{regency.name}: No valid NDVI pixels in either file")
                    continue

                monthly_ndvi = round(float(np.mean(available)), 4)

                existing = NDVIData.query.filter_by(
                    regency_id=regency.id, year=year, month=month
                ).first()

                if existing:
                    existing.ndvi_value = monthly_ndvi
                    existing.processing_date = datetime.utcnow()
                    existing.is_imputed = False
                    existing.data_source = 'modis'
                else:
                    db.session.add(NDVIData(
                        regency_id=regency.id,
                        year=year,
                        month=month,
                        ndvi_value=monthly_ndvi,
                        data_source='modis',
                        is_imputed=False
                    ))

                results['success'] += 1

            db.session.commit()

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
            db.session.rollback()
            return {
                'success': 0,
                'failed': 0,
                'errors': [f'Error processing NDVI files: {str(e)}']
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
                            'Precipitation': climate.precipitation if climate else None,
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
