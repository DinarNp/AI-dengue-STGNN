"""
CSV Data Management Service
Handles separate CSV files for dengue, climate, and NDVI data
Provides CRUD operations and data validation
"""
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from ..models import db, Regency, DengueCase, ClimateData, NDVIData


class CSVDataManager:
    """Manage data via CSV files with CRUD operations"""
    
    def __init__(self, config):
        self.config = config
        self.master_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'master'
        )
        self.training_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'training'
        )
        self.export_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'exports'
        )
        
        # Ensure folders exist
        os.makedirs(self.master_folder, exist_ok=True)
        os.makedirs(self.training_folder, exist_ok=True)
        os.makedirs(self.export_folder, exist_ok=True)
    
    # ==========================================
    # EXPORT FROM DATABASE TO CSV
    # ==========================================
    
    def export_dengue_cases(self, filename='dengue_cases.csv') -> Dict:
        """Export dengue cases from database to CSV"""
        try:
            cases = DengueCase.query.order_by(
                DengueCase.year, DengueCase.month, DengueCase.regency_id
            ).all()
            
            records = []
            for case in cases:
                records.append({
                    'Year': case.year,
                    'Month': case.month,
                    'Region': case.regency.name,
                    'Cases': case.cases,
                    'Data_Source': case.data_source,
                    'Notes': case.notes or ''
                })
            
            df = pd.DataFrame(records)
            filepath = os.path.join(self.master_folder, filename)
            df.to_csv(filepath, index=False)
            
            return {
                'success': True,
                'records': len(records),
                'filepath': filepath,
                'message': f'Exported {len(records)} dengue case records'
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def export_climate_data(self, filename='climate_data.csv') -> Dict:
        """Export climate data from database to CSV"""
        try:
            climate = ClimateData.query.order_by(
                ClimateData.year, ClimateData.month, ClimateData.regency_id
            ).all()
            
            records = []
            for c in climate:
                records.append({
                    'Year': c.year,
                    'Month': c.month,
                    'Region': c.regency.name,
                    'Temperature_Min': c.temperature_min,
                    'Temperature_Max': c.temperature_max,
                    'Temperature_Avg': c.temperature_avg,
                    'Humidity': c.humidity,
                    'Precipitation_Total': c.precipitation_total,
                    'Pressure': c.pressure,
                    'Wind_Speed': c.wind_speed,
                    'Wind_Direction': c.wind_direction,
                    'Cloud_Cover': c.cloud_cover,
                    'Data_Source': c.data_source
                })
            
            df = pd.DataFrame(records)
            filepath = os.path.join(self.master_folder, filename)
            df.to_csv(filepath, index=False)
            
            return {
                'success': True,
                'records': len(records),
                'filepath': filepath,
                'message': f'Exported {len(records)} climate data records'
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def export_ndvi_data(self, filename='ndvi_data.csv') -> Dict:
        """Export NDVI data from database to CSV"""
        try:
            ndvi = NDVIData.query.order_by(
                NDVIData.year, NDVIData.month, NDVIData.regency_id
            ).all()
            
            records = []
            for n in ndvi:
                records.append({
                    'Year': n.year,
                    'Month': n.month,
                    'Region': n.regency.name,
                    'NDVI': n.ndvi_value,
                    'Data_Source': n.data_source,
                    'Is_Imputed': 'Yes' if n.is_imputed else 'No'
                })
            
            df = pd.DataFrame(records)
            filepath = os.path.join(self.master_folder, filename)
            df.to_csv(filepath, index=False)
            
            return {
                'success': True,
                'records': len(records),
                'filepath': filepath,
                'message': f'Exported {len(records)} NDVI data records'
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def export_all_data(self) -> Dict:
        """Export all data types to separate CSV files"""
        results = {
            'dengue': self.export_dengue_cases(),
            'climate': self.export_climate_data(),
            'ndvi': self.export_ndvi_data()
        }
        
        success_count = sum(1 for r in results.values() if r['success'])
        
        return {
            'success': success_count == 3,
            'results': results,
            'message': f'Exported {success_count}/3 data types successfully'
        }
    
    # ==========================================
    # IMPORT FROM CSV TO DATABASE
    # ==========================================
    
    def import_dengue_cases(self, csv_path: str, user_id: int) -> Dict:
        """Import dengue cases from CSV to database"""
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required = ['Year', 'Month', 'Region', 'Cases']
            if not all(col in df.columns for col in required):
                return {
                    'success': False,
                    'message': f'Missing required columns. Need: {required}'
                }
            
            # Get regency mapping
            regencies = {r.name: r.id for r in Regency.query.all()}
            
            imported = 0
            updated = 0
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    if row['Region'] not in regencies:
                        errors.append(f"Row {idx+1}: Unknown regency '{row['Region']}'")
                        continue
                    
                    regency_id = regencies[row['Region']]
                    year = int(row['Year'])
                    month = int(row['Month'])
                    cases = int(row['Cases'])
                    notes = row.get('Notes', '')
                    
                    # Check if exists
                    existing = DengueCase.query.filter_by(
                        regency_id=regency_id,
                        year=year,
                        month=month
                    ).first()
                    
                    if existing:
                        existing.cases = cases
                        existing.notes = notes
                        existing.updated_at = datetime.utcnow()
                        updated += 1
                    else:
                        new_case = DengueCase(
                            regency_id=regency_id,
                            year=year,
                            month=month,
                            cases=cases,
                            notes=notes,
                            data_source='csv_import',
                            reported_by_id=user_id
                        )
                        db.session.add(new_case)
                        imported += 1
                    
                except Exception as e:
                    errors.append(f"Row {idx+1}: {str(e)}")
            
            db.session.commit()
            
            return {
                'success': True,
                'imported': imported,
                'updated': updated,
                'errors': errors,
                'message': f'Imported {imported}, Updated {updated}'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': str(e)}
    
    def import_climate_data(self, csv_path: str) -> Dict:
        """Import climate data from CSV to database"""
        try:
            df = pd.read_csv(csv_path)
            
            required = ['Year', 'Month', 'Region']
            if not all(col in df.columns for col in required):
                return {
                    'success': False,
                    'message': f'Missing required columns. Need: {required}'
                }
            
            regencies = {r.name: r.id for r in Regency.query.all()}
            
            imported = 0
            updated = 0
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    if row['Region'] not in regencies:
                        errors.append(f"Row {idx+1}: Unknown regency '{row['Region']}'")
                        continue
                    
                    regency_id = regencies[row['Region']]
                    year = int(row['Year'])
                    month = int(row['Month'])
                    
                    existing = ClimateData.query.filter_by(
                        regency_id=regency_id,
                        year=year,
                        month=month
                    ).first()
                    
                    climate_data = {
                        'temperature_min': float(row['Temperature_Min']) if pd.notna(row.get('Temperature_Min')) else None,
                        'temperature_max': float(row['Temperature_Max']) if pd.notna(row.get('Temperature_Max')) else None,
                        'temperature_avg': float(row['Temperature_Avg']) if pd.notna(row.get('Temperature_Avg')) else None,
                        'humidity': float(row['Humidity']) if pd.notna(row.get('Humidity')) else None,
                        'precipitation_total': float(row['Precipitation_Total']) if pd.notna(row.get('Precipitation_Total')) else None,
                        'pressure': float(row['Pressure']) if pd.notna(row.get('Pressure')) else None,
                        'wind_speed': float(row['Wind_Speed']) if pd.notna(row.get('Wind_Speed')) else None,
                        'wind_direction': float(row['Wind_Direction']) if pd.notna(row.get('Wind_Direction')) else None,
                        'cloud_cover': float(row['Cloud_Cover']) if pd.notna(row.get('Cloud_Cover')) else None,
                        'data_source': 'csv_import'
                    }
                    
                    if existing:
                        for key, value in climate_data.items():
                            setattr(existing, key, value)
                        updated += 1
                    else:
                        new_climate = ClimateData(
                            regency_id=regency_id,
                            year=year,
                            month=month,
                            **climate_data
                        )
                        db.session.add(new_climate)
                        imported += 1
                    
                except Exception as e:
                    errors.append(f"Row {idx+1}: {str(e)}")
            
            db.session.commit()
            
            return {
                'success': True,
                'imported': imported,
                'updated': updated,
                'errors': errors,
                'message': f'Imported {imported}, Updated {updated}'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': str(e)}
    
    def import_ndvi_data(self, csv_path: str) -> Dict:
        """Import NDVI data from CSV to database"""
        try:
            df = pd.read_csv(csv_path)
            
            required = ['Year', 'Month', 'Region', 'NDVI']
            if not all(col in df.columns for col in required):
                return {
                    'success': False,
                    'message': f'Missing required columns. Need: {required}'
                }
            
            regencies = {r.name: r.id for r in Regency.query.all()}
            
            imported = 0
            updated = 0
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    if row['Region'] not in regencies:
                        errors.append(f"Row {idx+1}: Unknown regency '{row['Region']}'")
                        continue
                    
                    regency_id = regencies[row['Region']]
                    year = int(row['Year'])
                    month = int(row['Month'])
                    ndvi_value = float(row['NDVI'])
                    is_imputed = row.get('Is_Imputed', 'No').lower() == 'yes'
                    
                    existing = NDVIData.query.filter_by(
                        regency_id=regency_id,
                        year=year,
                        month=month
                    ).first()
                    
                    if existing:
                        existing.ndvi_value = ndvi_value
                        existing.is_imputed = is_imputed
                        existing.data_source = 'csv_import'
                        updated += 1
                    else:
                        new_ndvi = NDVIData(
                            regency_id=regency_id,
                            year=year,
                            month=month,
                            ndvi_value=ndvi_value,
                            is_imputed=is_imputed,
                            data_source='csv_import'
                        )
                        db.session.add(new_ndvi)
                        imported += 1
                    
                except Exception as e:
                    errors.append(f"Row {idx+1}: {str(e)}")
            
            db.session.commit()
            
            return {
                'success': True,
                'imported': imported,
                'updated': updated,
                'errors': errors,
                'message': f'Imported {imported}, Updated {updated}'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': str(e)}
    
    # ==========================================
    # DATA MERGING FOR TRAINING
    # ==========================================
    
    def merge_data_for_training(self, year_start: int, year_end: int,
                                output_filename='training_data_merged.csv') -> Dict:
        """
        Merge dengue, climate, and NDVI data for training
        Creates combined CSV file with all features
        """
        try:
            records = []
            regencies = Regency.query.filter_by(is_active=True).all()
            
            missing_data = []
            
            for year in range(year_start, year_end + 1):
                for month in range(1, 13):
                    for regency in regencies:
                        # Get all data
                        dengue = DengueCase.query.filter_by(
                            regency_id=regency.id, year=year, month=month
                        ).first()
                        
                        climate = ClimateData.query.filter_by(
                            regency_id=regency.id, year=year, month=month
                        ).first()
                        
                        ndvi = NDVIData.query.filter_by(
                            regency_id=regency.id, year=year, month=month
                        ).first()
                        
                        # Check completeness
                        if not all([dengue, climate, ndvi]):
                            missing_data.append({
                                'year': year,
                                'month': month,
                                'regency': regency.name,
                                'missing': {
                                    'dengue': not dengue,
                                    'climate': not climate,
                                    'ndvi': not ndvi
                                }
                            })
                            continue
                        
                        # Create merged record
                        record = {
                            'Year': year,
                            'Region': regency.name,
                            'Month': month,
                            'Cases': dengue.cases,
                            'Latitude': regency.latitude,
                            'Longitude': regency.longitude,
                            'NDVI': ndvi.ndvi_value,
                            'Cloud_Cover': climate.cloud_cover,
                            'Humidity': climate.humidity,
                            'Precipitation_Total': climate.precipitation_total,
                            'Temperature_Min': climate.temperature_min,
                            'Temperature_Max': climate.temperature_max,
                            'Temperature_Avg': climate.temperature_avg,
                            'Pressure': climate.pressure,
                            'Wind_Speed': climate.wind_speed,
                            'Wind_Direction': climate.wind_direction
                        }
                        records.append(record)
            
            if len(records) == 0:
                return {
                    'success': False,
                    'message': 'No complete records found for the specified period',
                    'missing_data': missing_data
                }
            
            # Create DataFrame and save
            df = pd.DataFrame(records)
            output_path = os.path.join(self.training_folder, output_filename)
            df.to_csv(output_path, index=False)
            
            completeness = len(records) / (len(regencies) * (year_end - year_start + 1) * 12) * 100
            
            return {
                'success': True,
                'records': len(records),
                'output_path': output_path,
                'completeness': completeness,
                'missing_data': missing_data,
                'message': f'Merged {len(records)} complete records ({completeness:.1f}% complete)'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error merging data: {str(e)}'
            }
    
    # ==========================================
    # DATA VALIDATION
    # ==========================================
    
    def validate_data_completeness(self, year_start: int, year_end: int) -> Dict:
        """Check data completeness for training"""
        try:
            regencies = Regency.query.filter_by(is_active=True).all()
            total_expected = len(regencies) * (year_end - year_start + 1) * 12
            
            # Count available records
            dengue_count = DengueCase.query.filter(
                DengueCase.year >= year_start,
                DengueCase.year <= year_end
            ).count()
            
            climate_count = ClimateData.query.filter(
                ClimateData.year >= year_start,
                ClimateData.year <= year_end
            ).count()
            
            ndvi_count = NDVIData.query.filter(
                NDVIData.year >= year_start,
                NDVIData.year <= year_end
            ).count()
            
            return {
                'success': True,
                'total_expected': total_expected,
                'dengue_cases': {
                    'count': dengue_count,
                    'percentage': (dengue_count / total_expected * 100) if total_expected > 0 else 0
                },
                'climate_data': {
                    'count': climate_count,
                    'percentage': (climate_count / total_expected * 100) if total_expected > 0 else 0
                },
                'ndvi_data': {
                    'count': ndvi_count,
                    'percentage': (ndvi_count / total_expected * 100) if total_expected > 0 else 0
                },
                'overall_complete': min(dengue_count, climate_count, ndvi_count),
                'overall_percentage': (min(dengue_count, climate_count, ndvi_count) / total_expected * 100) if total_expected > 0 else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
