"""
Model Training Service
Integrates with existing STGNN training pipeline
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import json

# Add parent directory to path to import original training code
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from config.config import Config as ModelConfig
    from experiments.dengue_pipeline import DenguePredictionSystem
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("Warning: Training modules not available")

from ..models import db, Regency, DengueCase, ClimateData, NDVIData, ModelVersion, DataProcessingLog


class TrainingService:
    """Service for training STGNN models"""
    
    def __init__(self, config):
        self.config = config
        self.model_config = None
        if TRAINING_AVAILABLE:
            self.model_config = ModelConfig()
    
    def export_training_data(self, year_start: int, year_end: int, 
                            output_path: str) -> Dict:
        """
        Export data from database to CSV format for training
        
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
                        
                        # Only include if we have all data
                        if dengue and climate and ndvi:
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
                    'message': 'No complete data found for the specified period'
                }
            
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
    
    def train_model(self, data_csv_path: str, model_name: str, 
                   user_id: int, description: str = '') -> Dict:
        """
        Train a new STGNN model
        
        Args:
            data_csv_path: Path to training data CSV
            model_name: Name for the new model version
            user_id: ID of user initiating training
            description: Model description
            
        Returns:
            Dictionary with training results
        """
        if not TRAINING_AVAILABLE:
            return {
                'success': False,
                'message': 'Training modules not available. Please check installation.'
            }
        
        # Create processing log
        log = DataProcessingLog(
            user_id=user_id,
            process_type='model_training',
            status='started',
            details={
                'model_name': model_name,
                'data_file': data_csv_path
            }
        )
        db.session.add(log)
        db.session.commit()
        
        try:
            print(f"Starting training for model: {model_name}")
            print(f"Data file: {data_csv_path}")
            
            # Initialize prediction system with config
            prediction_system = DenguePredictionSystem(self.model_config)
            
            # Run training pipeline
            print("Running training pipeline...")
            model, metrics, info = prediction_system.run_complete_pipeline(
                data_path=data_csv_path
            )
            
            print(f"Training completed. Metrics: {metrics}")
            
            # Generate model filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_name.replace(' ', '_')}_{timestamp}.pth"
            model_path = os.path.join(self.config.MODEL_FOLDER, model_filename)
            
            # Save model
            import torch
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
            
            # Create model version record
            new_model_version = ModelVersion(
                version_name=model_name,
                model_file=model_path,
                description=description,
                mae=float(metrics.get('mae', 0)),
                rmse=float(metrics.get('rmse', 0)),
                r2_score=float(metrics.get('r2', 0)),
                training_data_file=data_csv_path,
                training_date=datetime.utcnow(),
                training_samples=info.get('total_samples', 0),
                is_active=False  # Don't auto-activate
            )
            db.session.add(new_model_version)
            
            # Update log
            log.status = 'completed'
            log.records_processed = info.get('total_samples', 0)
            log.completed_at = datetime.utcnow()
            log.details = {
                'model_name': model_name,
                'data_file': data_csv_path,
                'metrics': {
                    'mae': float(metrics.get('mae', 0)),
                    'rmse': float(metrics.get('rmse', 0)),
                    'r2': float(metrics.get('r2', 0))
                }
            }
            
            db.session.commit()
            
            return {
                'success': True,
                'model_id': new_model_version.id,
                'model_path': model_path,
                'metrics': {
                    'mae': float(metrics.get('mae', 0)),
                    'rmse': float(metrics.get('rmse', 0)),
                    'r2': float(metrics.get('r2', 0)),
                    'loss': float(metrics.get('loss', 0))
                },
                'training_samples': info.get('total_samples', 0),
                'message': f'Model "{model_name}" trained successfully'
            }
            
        except Exception as e:
            # Update log with error
            log.status = 'failed'
            log.error_message = str(e)
            log.completed_at = datetime.utcnow()
            db.session.commit()
            
            import traceback
            error_detail = traceback.format_exc()
            print(f"Training error: {error_detail}")
            
            return {
                'success': False,
                'message': f'Training failed: {str(e)}',
                'error_detail': error_detail
            }
    
    def get_training_status(self) -> Dict:
        """
        Get current training status and available models
        
        Returns:
            Dictionary with training status information
        """
        try:
            # Get all model versions
            models = ModelVersion.query.order_by(
                ModelVersion.training_date.desc()
            ).all()
            
            # Get active model
            active_model = ModelVersion.query.filter_by(is_active=True).first()
            
            # Get recent training logs
            recent_logs = DataProcessingLog.query.filter_by(
                process_type='model_training'
            ).order_by(DataProcessingLog.started_at.desc()).limit(5).all()
            
            return {
                'success': True,
                'total_models': len(models),
                'active_model': {
                    'id': active_model.id,
                    'name': active_model.version_name,
                    'mae': active_model.mae,
                    'rmse': active_model.rmse,
                    'r2_score': active_model.r2_score
                } if active_model else None,
                'recent_models': [
                    {
                        'id': m.id,
                        'name': m.version_name,
                        'date': m.training_date.strftime('%Y-%m-%d %H:%M') if m.training_date else 'N/A',
                        'mae': m.mae,
                        'is_active': m.is_active
                    } for m in models[:5]
                ],
                'recent_training_logs': [
                    {
                        'status': log.status,
                        'started': log.started_at.strftime('%Y-%m-%d %H:%M') if log.started_at else 'N/A',
                        'details': log.details
                    } for log in recent_logs
                ],
                'training_available': TRAINING_AVAILABLE
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error getting training status: {str(e)}'
            }
    
    def activate_model(self, model_id: int) -> Dict:
        """
        Activate a specific model version
        
        Args:
            model_id: ID of model to activate
            
        Returns:
            Dictionary with activation result
        """
        try:
            # Deactivate all models
            ModelVersion.query.update({'is_active': False})
            
            # Activate selected model
            model = ModelVersion.query.get(model_id)
            if not model:
                return {
                    'success': False,
                    'message': 'Model not found'
                }
            
            model.is_active = True
            db.session.commit()
            
            return {
                'success': True,
                'message': f'Model "{model.version_name}" activated successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': f'Error activating model: {str(e)}'
            }
