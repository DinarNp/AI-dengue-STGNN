"""
Prediction Service
Integrates with the existing STGNN model for dengue prediction
"""
import sys
import os
import math
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
                          window_size: int = 4) -> tuple:
        """
        Prepare input data for prediction.
        Requires historical data for the past window_size months.

        For months where actual dengue cases are missing, falls back to the
        Prediction table (used when chaining multi-step future predictions).
        For months where climate/NDVI data are missing, falls back to the
        same calendar month from the previous year.

        Returns:
            (DataFrame, None)       on success
            (None, error_message)   on failure
        """
        try:
            regency = Regency.query.get(regency_id)
            if not regency:
                return None, 'Regency not found'

            # Calculate start month for historical window
            start_year = year
            start_month = month - window_size

            while start_month <= 0:
                start_month += 12
                start_year -= 1

            records = []
            current_year = start_year
            current_month = start_month

            for _ in range(window_size):
                period = f"{current_year}-{current_month:02d}"

                # ── Dengue cases ──────────────────────────────────────────
                dengue = DengueCase.query.filter_by(
                    regency_id=regency_id,
                    year=current_year,
                    month=current_month
                ).first()

                if dengue:
                    cases_value = dengue.cases
                else:
                    # Fall back to already-generated prediction for this month
                    pred_fallback = Prediction.query.filter_by(
                        regency_id=regency_id,
                        year=current_year,
                        month=current_month
                    ).first()
                    if pred_fallback:
                        cases_value = pred_fallback.predicted_cases
                    else:
                        return None, (
                            f"Missing dengue data for {period}. "
                            "Generate the prediction for that month first, "
                            "or upload the actual case data."
                        )

                # ── Climate data ──────────────────────────────────────────
                climate = ClimateData.query.filter_by(
                    regency_id=regency_id,
                    year=current_year,
                    month=current_month
                ).first()

                if not climate:
                    # Fall back to same month of previous year
                    climate = ClimateData.query.filter_by(
                        regency_id=regency_id,
                        year=current_year - 1,
                        month=current_month
                    ).first()

                if not climate:
                    return None, (
                        f"Missing climate data for {period} "
                        f"(and no previous-year fallback for month {current_month:02d})."
                    )

                # ── NDVI data ─────────────────────────────────────────────
                ndvi = NDVIData.query.filter_by(
                    regency_id=regency_id,
                    year=current_year,
                    month=current_month
                ).first()

                if not ndvi:
                    # Fall back to same month of previous year
                    ndvi = NDVIData.query.filter_by(
                        regency_id=regency_id,
                        year=current_year - 1,
                        month=current_month
                    ).first()

                if not ndvi:
                    return None, (
                        f"Missing NDVI data for {period} "
                        f"(and no previous-year fallback for month {current_month:02d})."
                    )

                records.append({
                    'Year': current_year,
                    'Month': current_month,
                    'Region': regency.name,
                    'Latitude': regency.latitude,
                    'Longitude': regency.longitude,
                    'Cases': cases_value,
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
                })

                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1

            return pd.DataFrame(records), None

        except Exception as e:
            print(f"Error preparing input data: {str(e)}")
            return None, str(e)
    
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
            input_data, data_error = self.prepare_input_data(regency_id, year, month)

            if input_data is None:
                return {
                    'success': False,
                    'message': data_error or (
                        f'Insufficient historical data. Need {self.config.get("WINDOW_SIZE_MONTHLY", 4)} months of data.'
                    )
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
            
            # Determine risk level (alert/no_alert) via endemic channel
            risk_info = self._calculate_risk_level(predicted_cases, regency_id, month)
            risk_level        = risk_info['risk_level']
            alert_threshold   = risk_info['alert_threshold']
            hist_mean         = risk_info['hist_mean']
            hist_sd           = risk_info['hist_sd']
            exceed_probability = risk_info['exceed_probability']

            # Probability of cases occurring = 1 - P(zero cases)
            case_probability = max(0.0, 1.0 - zero_prob)

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
                existing.alert_threshold = alert_threshold
                existing.hist_mean = hist_mean
                existing.hist_sd = hist_sd
                existing.exceed_probability = exceed_probability
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
                    alert_threshold=alert_threshold,
                    hist_mean=hist_mean,
                    hist_sd=hist_sd,
                    exceed_probability=exceed_probability,
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
                'case_probability': round(case_probability * 100, 1),
                'confidence_lower': round(confidence_lower, 1),
                'confidence_upper': round(confidence_upper, 1),
                'risk_level': risk_level,
                'alert_threshold': alert_threshold,
                'hist_mean': hist_mean,
                'hist_sd': hist_sd,
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
    
    def predict_next_n_months(self, from_year: int, from_month: int, n: int = 3) -> Dict:
        """
        Generate predictions for the next n months starting from the month
        AFTER (from_year, from_month).

        Each month is predicted in order so that the saved prediction for
        month T can serve as the dengue-cases fallback for month T+1 when
        actual case data does not yet exist.

        Args:
            from_year:  Year of the last month with actual data
            from_month: Month of the last month with actual data
            n:          Number of future months to predict (default 3)

        Returns:
            Dictionary with per-month results
        """
        month_results = []
        current_year = from_year
        current_month = from_month

        for _ in range(n):
            # Advance to the next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

            result = self.predict_all_regencies(current_year, current_month)
            month_results.append({
                'year': current_year,
                'month': current_month,
                'result': result
            })

        total_success = sum(r['result']['success'] for r in month_results)
        total_predictions = sum(
            len(r['result'].get('predictions', [])) for r in month_results
        )

        return {
            'success': True,
            'months_processed': len(month_results),
            'total_predictions': total_predictions,
            'results': month_results,
            'message': (
                f'Generated {total_predictions} predictions '
                f'for {len(month_results)} months'
            )
        }

    def _calculate_risk_level(self, predicted_cases: float, regency_id: int, month: int) -> dict:
        """
        Calculate alert status based on endemic channel method.
        Threshold = historical mean + 1.25 * SD for the same regency+month.

        Returns dict with keys: risk_level, alert_threshold, hist_mean, hist_sd
        """
        historical = db.session.query(DengueCase.cases).filter(
            DengueCase.regency_id == regency_id,
            DengueCase.month == month
        ).all()

        values = [r.cases for r in historical if r.cases is not None]

        if len(values) >= 3:
            hist_mean = float(np.mean(values))
            hist_sd   = float(np.std(values, ddof=1))
        else:
            # Fallback: no enough history — use flat threshold
            hist_mean = float(np.mean(values)) if values else 0.0
            hist_sd   = 0.0

        alert_threshold = hist_mean + 1.25 * hist_sd
        risk_level = 'alert' if predicted_cases > alert_threshold else 'no_alert'

        # P(actual > threshold | actual ~ N(predicted, hist_sd))
        # = 1 - Φ((threshold - predicted) / hist_sd)
        # using math.erfc: 1 - Φ(z) = 0.5 * erfc(z / sqrt(2))
        if hist_sd > 0:
            z = (alert_threshold - predicted_cases) / hist_sd
            exceed_prob = 0.5 * math.erfc(z / math.sqrt(2))
        else:
            exceed_prob = 1.0 if predicted_cases > alert_threshold else 0.0

        return {
            'risk_level': risk_level,
            'alert_threshold': round(alert_threshold, 2),
            'hist_mean': round(hist_mean, 2),
            'hist_sd': round(hist_sd, 2),
            'exceed_probability': round(exceed_prob * 100, 1),
        }
    
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
            'no_alert': (
                "Status: NO ALERT. Maintain routine dengue prevention activities:\n"
                "- Continuation of routine vector control and household inspections\n"
                "- Regular health promotion and school-based education\n"
                "- Surveillance system maintenance for early warning detection\n"
                "- Seasonal preparedness activities during pre-wet season transition\n"
                "- Community engagement in long-term environmental management"
            ),
            'alert': (
                "Status: ALERT. Predicted cases exceed endemic threshold — activate response:\n"
                "- Intensify vector control (larval source management, fogging in high-density areas)\n"
                "- Deploy rapid response teams for active case detection and contact tracing\n"
                "- Strengthen community-based surveillance through trained volunteers\n"
                "- Escalate health promotion across all channels (radio, social media, community meetings)\n"
                "- Coordinate healthcare facilities for case management readiness and resource stockpiling"
            ),
        }

        return recommendations.get(risk_level, "No recommendation available")