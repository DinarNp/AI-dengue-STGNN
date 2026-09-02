"""
Prediction Service (weekly cadence)

Builds real (window_size, n_nodes, n_features) weekly input tensors from the
database and runs them through the canonical revision_experiments STGNN via
DenguePredictor. Feature engineering (lag features, cyclical encodings,
scaling) reuses DengueDataPreprocessor directly rather than reimplementing
it, so live inference is guaranteed to match how the model was trained.
"""
import math
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

REVISION_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    'revision_experiments',
)
if REVISION_EXPERIMENTS not in sys.path:
    sys.path.insert(0, REVISION_EXPERIMENTS)

try:
    from data.preprocessor import DengueDataPreprocessor
    from config.config import Config as ModelConfig
    from .dengue_predictor import DenguePredictor
    PREDICTION_AVAILABLE = True
except ImportError as e:
    PREDICTION_AVAILABLE = False
    print(f"Warning: STGNN model not available ({e}). Predictions will use fallback.")

HISTORY_MULTIPLIER = 1.25  # matches webapp_restructured/app/services/prediction.py::_calculate_risk_level

from epiweeks import Week

from ..models import db, Regency, DengueCase, ClimateData, NDVIData, Prediction, ModelVersion


class PredictionService:
    """Service for generating weekly dengue predictions using the canonical STGNN model."""

    def __init__(self, config):
        self.config = config
        self.model: Optional['DenguePredictor'] = None
        self.model_version = None
        self._preprocessor = DengueDataPreprocessor(ModelConfig()) if PREDICTION_AVAILABLE else None
        self.load_active_model()

    def load_active_model(self):
        try:
            active_model = ModelVersion.query.filter_by(is_active=True).first()
            if active_model and PREDICTION_AVAILABLE:
                self.model = DenguePredictor(active_model.model_file)
                self.model_version = active_model.version_name
                print(f"Loaded model: {active_model.version_name}")
                return True
            print("No active model found or prediction module unavailable")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def _region_dataframe(self, regency: Regency, upto_epi_year: int, upto_epi_week: int) -> pd.DataFrame:
        """All historical weekly rows for one region, from the earliest record up to
        and including (upto_epi_year, upto_epi_week), as a raw (unengineered) DataFrame
        matching the canonical CSV schema."""
        rows = []
        cases = {(c.epi_year, c.epi_week): c.cases for c in DengueCase.query.filter_by(regency_id=regency.id)}
        climate = {(c.epi_year, c.epi_week): c for c in ClimateData.query.filter_by(regency_id=regency.id)}
        ndvi = {(n.epi_year, n.epi_week): n for n in NDVIData.query.filter_by(regency_id=regency.id)}

        all_weeks = sorted(set(cases) | set(climate) | set(ndvi))
        all_weeks = [w for w in all_weeks if w <= (upto_epi_year, upto_epi_week)]

        for epi_year, epi_week in all_weeks:
            cl = climate.get((epi_year, epi_week))
            nd = ndvi.get((epi_year, epi_week))
            if cl is None or nd is None:
                continue  # can't build a feature row without climate+NDVI for that week
            rows.append({
                'Year': epi_year, 'Week': epi_week, 'Region': regency.name,
                'Latitude': regency.latitude, 'Longitude': regency.longitude,
                'Cases': cases.get((epi_year, epi_week), 0),
                'NDVI': nd.ndvi_value,
                'Cloud_Cover': cl.cloud_cover, 'Humidity': cl.humidity,
                'Precipitation_Total': cl.precipitation_total,
                'Temperature_Min': cl.temperature_min, 'Temperature_Max': cl.temperature_max,
                'Temperature_Avg': cl.temperature_avg, 'Pressure': cl.pressure,
                'Wind_Speed': cl.wind_speed, 'Wind_Direction': cl.wind_direction,
            })
        return pd.DataFrame(rows)

    def prepare_input_data(self, target_epi_year: int, target_epi_week: int) -> tuple:
        """
        Build a (window_size, n_nodes, n_features) tensor, one row per week of
        the input window ending `forecast_horizon` weeks before the target
        week, across all regions in the model's node order.

        Returns:
            (np.ndarray, None)      on success
            (None, error_message)   on failure
        """
        if not (self.model and PREDICTION_AVAILABLE):
            return None, 'Prediction model not available'

        window_size = self.model.window_size
        horizon = self.model.forecast_horizon
        target_week = Week(target_epi_year, target_epi_week, system='iso')
        last_input_week = target_week - horizon
        first_input_week = last_input_week - (window_size - 1)

        feature_cols = self.model.metadata.get('feature_cols', [])
        scaler = self.model.metadata.get('scaler')
        if not feature_cols or scaler is None:
            return None, 'Model checkpoint is missing feature_cols/scaler metadata'

        node_arrays = []
        for node_id in self.model.node_ids:
            regency = Regency.query.filter_by(name=node_id).first()
            if regency is None:
                return None, f"Region '{node_id}' (expected by the model) not found in database"

            raw_df = self._region_dataframe(regency, last_input_week.year, last_input_week.week)
            if raw_df.empty:
                return None, f"No historical data available for {regency.name} up to {last_input_week}"

            engineered = self._preprocessor.create_date_features(raw_df)
            engineered = self._preprocessor.create_lag_features(engineered)
            engineered = self._preprocessor.handle_missing_values(engineered)
            engineered['Kecamatan_encoded'] = 0

            mask = engineered.apply(
                lambda r: first_input_week.year <= r['Year'] <= last_input_week.year and
                          Week(int(r['Year']), int(r['Week']), system='iso') >= first_input_week and
                          Week(int(r['Year']), int(r['Week']), system='iso') <= last_input_week,
                axis=1,
            )
            window_df = engineered[mask].sort_values(['Year', 'Week'])

            if len(window_df) != window_size:
                return None, (
                    f"Insufficient contiguous history for {regency.name}: need {window_size} weeks "
                    f"ending {last_input_week}, found {len(window_df)}. Missing weeks in the range "
                    f"must be reported/fetched before a prediction can be made."
                )

            for col in feature_cols:
                if col not in window_df.columns:
                    window_df[col] = 0
            scaled = scaler.transform(window_df[feature_cols])
            node_arrays.append(scaled)

        # (n_nodes, window_size, n_features) -> (window_size, n_nodes, n_features)
        window_tensor = np.stack(node_arrays, axis=0).transpose(1, 0, 2)
        return window_tensor, None

    def predict_single_regency(self, regency_id: int, epi_year: int, epi_week: int) -> Dict:
        regency = Regency.query.get(regency_id)
        if not regency:
            return {'success': False, 'message': 'Regency not found'}

        window_tensor, data_error = self.prepare_input_data(epi_year, epi_week)
        if window_tensor is None:
            return {'success': False, 'message': data_error}

        try:
            result = self.model.predict_with_all_locations(window_tensor, regency.name)
        except Exception as e:
            return {'success': False, 'message': f'Model inference error: {e}'}

        predicted_cases = result['predicted_cases']
        zero_prob = result['zero_probability']
        risk_info = self._calculate_risk_level(predicted_cases, regency_id, epi_year, epi_week)
        case_probability = max(0.0, 1.0 - zero_prob)
        confidence_lower = max(0, predicted_cases * 0.7)
        confidence_upper = predicted_cases * 1.3

        existing = Prediction.query.filter_by(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week).first()
        if existing is None:
            existing = Prediction(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week)
            db.session.add(existing)
        existing.predicted_cases = predicted_cases
        existing.zero_probability = zero_prob
        existing.confidence_lower = confidence_lower
        existing.confidence_upper = confidence_upper
        existing.risk_level = risk_info['risk_level']
        existing.alert_threshold = risk_info['alert_threshold']
        existing.hist_mean = risk_info['hist_mean']
        existing.hist_sd = risk_info['hist_sd']
        existing.exceed_probability = risk_info['exceed_probability']
        existing.model_version = self.model_version
        existing.prediction_date = datetime.utcnow()
        db.session.commit()

        return {
            'success': True,
            'regency': regency.name,
            'epi_year': epi_year,
            'epi_week': epi_week,
            'predicted_cases': round(predicted_cases, 1),
            'zero_probability': round(zero_prob * 100, 1),
            'case_probability': round(case_probability * 100, 1),
            'confidence_lower': round(confidence_lower, 1),
            'confidence_upper': round(confidence_upper, 1),
            'risk_level': risk_info['risk_level'],
            'alert_threshold': risk_info['alert_threshold'],
            'hist_mean': risk_info['hist_mean'],
            'hist_sd': risk_info['hist_sd'],
        }

    def _calculate_risk_level(self, predicted_cases: float, regency_id: int, epi_year: int, epi_week: int) -> dict:
        """
        Statistical "endemic channel" method, matching
        webapp_restructured/app/services/prediction.py::_calculate_risk_level,
        adapted from calendar-month to epidemiological-week grouping.

        Threshold = historical mean + 1.25 * SD of this region's actual cases in
        the SAME epi-week across prior years (2021 up to epi_year - 1, so the
        current/target year is never included). Binary classes only.
        """
        historical = DengueCase.query.filter(
            DengueCase.regency_id == regency_id,
            DengueCase.epi_week == epi_week,
            DengueCase.epi_year >= 2021,
            DengueCase.epi_year <= epi_year - 1,
        ).all()

        values = [c.cases for c in historical if c.cases is not None]

        if len(values) >= 2:
            hist_mean = float(np.mean(values))
            hist_sd = float(np.std(values, ddof=1))
        else:
            hist_mean = float(np.mean(values)) if values else 0.0
            hist_sd = 0.0

        alert_threshold = hist_mean + HISTORY_MULTIPLIER * hist_sd
        risk_level = 'alert' if predicted_cases > alert_threshold else 'no_alert'

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

    def predict_all_regencies(self, epi_year: int, epi_week: int) -> Dict:
        results = {'success': 0, 'failed': 0, 'predictions': [], 'errors': []}
        for regency in Regency.query.filter_by(is_active=True).all():
            result = self.predict_single_regency(regency.id, epi_year, epi_week)
            if result['success']:
                results['success'] += 1
                results['predictions'].append(result)
            else:
                results['failed'] += 1
                results['errors'].append(f"{regency.name}: {result['message']}")
        return results

    def predict_next_n_weeks(self, from_epi_year: int, from_epi_week: int, n_weeks: int) -> Dict:
        """Generate predictions for all regencies for each of the next n_weeks
        epi-weeks after (from_epi_year, from_epi_week)."""
        wk = Week(from_epi_year, from_epi_week, system='iso')
        results = []
        total_success = 0
        total_predictions = 0
        for _ in range(n_weeks):
            wk = wk + 1
            week_result = self.predict_all_regencies(wk.year, wk.week)
            results.append({
                'epi_year': wk.year,
                'epi_week': wk.week,
                'result': week_result,
            })
            total_success += week_result['success']
            total_predictions += len(week_result['predictions'])
        return {
            'success': True,
            'total_predictions': total_predictions,
            'weeks_processed': len(results),
            'results': results,
            'message': f'Generated {total_predictions} predictions for {len(results)} weeks',
        }

    def get_recommendation(self, predicted_cases: float, risk_level: str) -> str:
        """Recommendation text per risk status, matching
        webapp_restructured/app/services/prediction.py::get_recommendation."""
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
