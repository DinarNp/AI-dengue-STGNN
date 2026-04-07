"""
Prediction Service
Integrates with the existing STGNN model for dengue prediction
"""
import sys
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path to import from original codebase
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from models.predictor import DenguePredictor
    from config.config import Config as ModelConfig
    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False
    print("Warning: STGNN model not available. Predictions will use fallback.")

from ..models import db, Regency, DengueCase, ClimateData, NDVIData, Prediction, ModelVersion


class PredictionService:
    """Service for generating dengue predictions using STGNN model"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_version = None
        
        # Load active model
        self.load_active_model()
    
    def load_active_model(self):
        """Load the active model from database"""
        try:
            active_model = ModelVersion.query.filter_by(is_active=True).first()
            
            if active_model and PREDICTION_AVAILABLE:
                self.model = DenguePredictor(active_model.model_file)
                self.model_version = active_model.version_name
                print(f"Loaded model: {active_model.version_name}")
                return True
            else:
                print("No active model found or prediction module unavailable")
                return False
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def prepare_input_data(self, regency_id: int, year: int, month: int, 
                          window_size: int = 4) -> Optional[pd.DataFrame]:
        """
        Prepare input data for prediction
        Requires historical data for the past window_size months
        
        Args:
            regency_id: ID of the regency
            year: Target year
            month: Target month
            window_size: Number of historical months needed
            
        Returns:
            DataFrame with features or None if insufficient data
        """
        try:
            regency = Regency.query.get(regency_id)
            if not regency:
                return None
            
            # Calculate start month for historical window
            start_year = year
            start_month = month - window_size
            
            while start_month <= 0:
                start_month += 12
                start_year -= 1
            
            # Collect historical data
            records = []
            
            current_year = start_year
            current_month = start_month
            
            for _ in range(window_size):
                # Get dengue cases
                dengue = DengueCase.query.filter_by(
                    regency_id=regency_id,
                    year=current_year,
                    month=current_month
                ).first()
                
                # Get climate data
                climate = ClimateData.query.filter_by(
                    regency_id=regency_id,
                    year=current_year,
                    month=current_month
                ).first()
                
                # Get NDVI data
                ndvi = NDVIData.query.filter_by(
                    regency_id=regency_id,
                    year=current_year,
                    month=current_month
                ).first()
                
                if not dengue or not climate or not ndvi:
                    # Insufficient historical data
                    return None
                
                record = {
                    'Year': current_year,
                    'Month': current_month,
                    'Region': regency.name,
                    'Latitude': regency.latitude,
                    'Longitude': regency.longitude,
                    'Cases': dengue.cases,
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
                
                # Move to next month
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
            
            df = pd.DataFrame(records)
            return df
            
        except Exception as e:
            print(f"Error preparing input data: {str(e)}")
            return None
    
    def predict_single_regency(self, regency_id: int, year: int, month: int) -> Dict:
        """
        Generate prediction for a single regency
        
        Args:
            regency_id: ID of the regency
            year: Target year
            month: Target month
            
        Returns:
            Dictionary with prediction results
        """
        try:
            regency = Regency.query.get(regency_id)
            if not regency:
                return {
                    'success': False,
                    'message': 'Regency not found'
                }
            
            # Prepare input data
            input_data = self.prepare_input_data(regency_id, year, month)
            
            if input_data is None:
                return {
                    'success': False,
                    'message': f'Insufficient historical data. Need {self.config.WINDOW_SIZE_MONTHLY} months of data.'
                }
            
            # Make prediction
            if self.model and PREDICTION_AVAILABLE:
                result = self.model.predict(input_data)
                
                predicted_cases = float(result['predictions'][0][0])
                zero_prob = float(result.get('zero_probabilities', [[0]])[0][0])
            else:
                # Fallback: simple moving average
                predicted_cases = float(input_data['Cases'].tail(3).mean())
                zero_prob = 0.0
            
            # Determine risk level
            risk_level = self._calculate_risk_level(predicted_cases)
            
            # Calculate confidence bounds (simple approach: ±30%)
            confidence_lower = max(0, predicted_cases * 0.7)
            confidence_upper = predicted_cases * 1.3
            
            # Save prediction to database
            existing = Prediction.query.filter_by(
                regency_id=regency_id,
                year=year,
                month=month
            ).first()
            
            if existing:
                existing.predicted_cases = predicted_cases
                existing.zero_probability = zero_prob
                existing.confidence_lower = confidence_lower
                existing.confidence_upper = confidence_upper
                existing.risk_level = risk_level
                existing.model_version = self.model_version
                existing.prediction_date = datetime.utcnow()
            else:
                new_pred = Prediction(
                    regency_id=regency_id,
                    year=year,
                    month=month,
                    predicted_cases=predicted_cases,
                    zero_probability=zero_prob,
                    confidence_lower=confidence_lower,
                    confidence_upper=confidence_upper,
                    risk_level=risk_level,
                    model_version=self.model_version
                )
                db.session.add(new_pred)
            
            db.session.commit()
            
            return {
                'success': True,
                'regency': regency.name,
                'year': year,
                'month': month,
                'predicted_cases': round(predicted_cases, 1),
                'zero_probability': round(zero_prob * 100, 1),
                'confidence_lower': round(confidence_lower, 1),
                'confidence_upper': round(confidence_upper, 1),
                'risk_level': risk_level
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': f'Error generating prediction: {str(e)}'
            }
    
    def predict_all_regencies(self, year: int, month: int) -> Dict:
        """
        Generate predictions for all active regencies
        
        Args:
            year: Target year
            month: Target month
            
        Returns:
            Dictionary with aggregated results
        """
        results = {
            'success': 0,
            'failed': 0,
            'predictions': [],
            'errors': []
        }
        
        try:
            regencies = Regency.query.filter_by(is_active=True).all()
            
            for regency in regencies:
                result = self.predict_single_regency(regency.id, year, month)
                
                if result['success']:
                    results['success'] += 1
                    results['predictions'].append(result)
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{regency.name}: {result['message']}")
            
            return results
            
        except Exception as e:
            return {
                'success': 0,
                'failed': len(Regency.query.filter_by(is_active=True).all()),
                'predictions': [],
                'errors': [str(e)]
            }
    
    def _calculate_risk_level(self, predicted_cases: float) -> str:
        """
        Calculate risk level based on predicted cases
        
        Args:
            predicted_cases: Number of predicted cases
            
        Returns:
            Risk level string
        """
        thresholds = self.config.RISK_THRESHOLDS
        
        if predicted_cases < thresholds['low']:
            return 'low'
        elif predicted_cases < thresholds['medium']:
            return 'medium'
        elif predicted_cases < thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def get_recommendation(self, predicted_cases: float, risk_level: str) -> str:
        """
        Generate health recommendation based on prediction
        
        Args:
            predicted_cases: Number of predicted cases
            risk_level: Risk level
            
        Returns:
            Recommendation text
        """
        recommendations = {
            'low': (
                "Risk level is LOW. Continue routine dengue prevention activities:\n"
                "- Maintain regular fogging schedules\n"
                "- Continue public health education\n"
                "- Monitor mosquito breeding sites weekly"
            ),
            'medium': (
                "Risk level is MEDIUM. Increase prevention efforts:\n"
                "- Intensify fogging in high-risk areas\n"
                "- Conduct community campaigns about 3M (Drain, Cover, Bury)\n"
                "- Increase surveillance for fever cases\n"
                "- Distribute mosquito repellent to vulnerable populations"
            ),
            'high': (
                "Risk level is HIGH. Implement enhanced control measures:\n"
                "- Daily fogging in affected areas\n"
                "- Mobilize health teams for active case finding\n"
                "- Set up fever clinics in community health centers\n"
                "- Coordinate with local government for emergency response\n"
                "- Distribute educational materials door-to-door"
            ),
            'very_high': (
                "Risk level is VERY HIGH. Activate emergency response:\n"
                "- Declare dengue outbreak status\n"
                "- Implement mass fogging campaign\n"
                "- Deploy additional medical staff to health centers\n"
                "- Coordinate with hospitals for bed availability\n"
                "- Activate rapid response team\n"
                "- Intensive media campaign for public awareness\n"
                "- Request provincial/national support if needed"
            )
        }
        
        return recommendations.get(risk_level, "No recommendation available")
