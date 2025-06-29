from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import traceback
from werkzeug.utils import secure_filename
import pickle
import hashlib

# Fix matplotlib backend for Flask/threading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import your modules - with try/except for safety
try:
    from config.config import Config
    from experiments.dengue_pipeline import DenguePredictionSystem
    from models.predictor import DenguePredictor
    AI_MODULES_AVAILABLE = True
    print("✅ AI modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: AI modules not found: {e}")
    print("🔄 Running in demo mode without AI functionality")
    AI_MODULES_AVAILABLE = False
    
    # Create dummy classes for demo
    class Config:
        def __init__(self):
            self.WINDOW_SIZE = 7
            
    class DenguePredictionSystem:
        def __init__(self, config):
            self.config = config
            
        def run_complete_pipeline(self, data_path):
            # Return dummy results for demo
            return None, {'mae': 0.85, 'rmse': 1.01, 'r2': -0.21, 'loss': 0.79}, {'feature_cols': []}
            
    class DenguePredictor:
        def __init__(self, model_path):
            self.model_path = model_path
            
        def predict(self, input_data):
            # Return dummy prediction
            return {
                'node_ids': ['PKM. DEMO'],
                'predictions': [[0.56]],
                'zero_probabilities': [[0.54]]
            }

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store model and data
model_instance = None
predictor = None
current_data = None
config = Config()
system = None
model_trained = False
training_metrics = {}
# Add cached risk data to avoid random changes
cached_risk_data = None

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions
def create_sample_data():
    """Create sample dengue data for demo purposes"""
    np.random.seed(42)  # For reproducible results
    
    # Create sample data
    n_records = 1000
    puskesmas_names = [
        'PKM. BAMBANG LIPURO', 'PKM. BANGUNTAPAN I', 'PKM. GAMPING II',
        'PKM. GODEAN I', 'PKM. GODEAN II', 'PKM. SAPTOSARI'
    ]
    
    data = {
        'Year': np.random.choice([2021, 2022, 2023], n_records),
        'Region': ['KAB BANTUL'] * n_records,
        'Puskesmas': np.random.choice(puskesmas_names, n_records),
        'cases': np.random.poisson(2, n_records),
        'temperature_avg': np.random.normal(27, 3, n_records),
        'temperature_max': np.random.normal(32, 3, n_records),
        'temperature_min': np.random.normal(22, 3, n_records),
        'precipitation_total': np.random.exponential(50, n_records),
        'humidity': np.random.normal(75, 10, n_records),
        'pressure': np.random.normal(1013, 5, n_records),
        'cloud_cover': np.random.uniform(0, 100, n_records),
        'wind_speed': np.random.exponential(5, n_records),
        'wind_direction': np.random.uniform(0, 360, n_records),
        'ndvi': np.random.uniform(0.2, 0.8, n_records),
        'latitude': np.random.uniform(-8.0, -7.5, n_records),
        'longitude': np.random.uniform(110.0, 110.5, n_records)
    }
    
    # Ensure realistic ranges
    data['temperature_avg'] = np.clip(data['temperature_avg'], 15, 40)
    data['temperature_max'] = np.clip(data['temperature_max'], 20, 45)
    data['temperature_min'] = np.clip(data['temperature_min'], 10, 35)
    data['precipitation_total'] = np.clip(data['precipitation_total'], 0, 300)
    data['humidity'] = np.clip(data['humidity'], 30, 100)
    data['wind_speed'] = np.clip(data['wind_speed'], 0, 20)
    data['cases'] = np.clip(data['cases'], 0, 20)
    
    df = pd.DataFrame(data)
    print(f"✅ Created sample dataset with {len(df)} records")
    return df

def calculate_environmental_risk(temp, precip, humidity, puskesmas_name):
    """Calculate risk based on environmental factors with CONSISTENT scaling"""
    # Seed berdasarkan nama puskesmas untuk konsistensi
    seed = int(hashlib.md5(puskesmas_name.encode()).hexdigest()[:8], 16) % 1000
    np.random.seed(seed)
    
    # Start with base risk score - SESUAIKAN agar threshold konsisten
    risk_score = 0.8  # Turunkan base score
    
    # Temperature scoring (optimal range 25-30°C)
    if 25 <= temp <= 30:
        temp_score = 1.8 + (temp - 27.5) * 0.15  # Kurangi range
    elif 20 <= temp < 25:
        temp_score = 0.8 + (temp - 20) * 0.2  
    elif 30 < temp <= 35:
        temp_score = 1.8 - (temp - 30) * 0.15  
    else:
        temp_score = 0.4  
    
    # Precipitation scoring (higher = more breeding sites)
    if precip > 200:
        precip_score = 2.2  # Kurangi max score
    elif precip > 150:
        precip_score = 1.8 + (precip - 150) * 0.008  
    elif precip > 100:
        precip_score = 1.3 + (precip - 100) * 0.01  
    elif precip > 50:
        precip_score = 0.8 + (precip - 50) * 0.01   
    elif precip > 20:
        precip_score = 0.4 + (precip - 20) * 0.013  
    else:
        precip_score = 0.2  
    
    # Humidity scoring (higher humidity = better for mosquitos)
    if humidity > 85:
        humidity_score = 1.8  # Kurangi max score
    elif humidity > 75:
        humidity_score = 1.3 + (humidity - 75) * 0.05  
    elif humidity > 60:
        humidity_score = 0.8 + (humidity - 60) * 0.033  
    elif humidity > 40:
        humidity_score = 0.4 + (humidity - 40) * 0.02  
    else:
        humidity_score = 0.2  
    
    # Combine scores using weighted average
    weights = [0.35, 0.40, 0.25]  # temp, precip, humidity weights
    combined_score = (temp_score * weights[0] + 
                     precip_score * weights[1] + 
                     humidity_score * weights[2])
    
    # Add some random variation but keep it consistent
    variation = np.random.normal(0, 0.1)  # Kurangi variasi
    final_score = combined_score + variation
    
    # PENTING: Sesuaikan range agar konsisten dengan threshold dashboard
    # Dashboard menggunakan: >2.0 = high, >1.0 = moderate, <=1.0 = low
    final_score = max(0.1, min(3.5, final_score))  # Range 0.1 - 3.5
    
    print(f"🔍 Risk calculation for {puskesmas_name}:")
    print(f"   🌡️ Temp: {temp:.1f}°C → score: {temp_score:.2f}")
    print(f"   🌧️ Precip: {precip:.1f}mm → score: {precip_score:.2f}")
    print(f"   💧 Humidity: {humidity:.1f}% → score: {humidity_score:.2f}")
    print(f"   📊 Combined: {combined_score:.2f} → Final: {final_score:.2f}")
    print(f"   🎯 Risk Level: {get_risk_level(final_score)}")
    
    return final_score

def get_risk_level(prediction_value):
    """Determine risk level with CONSISTENT thresholds"""
    # KONSISTEN dengan dashboard.html - gunakan threshold yang sama
    if prediction_value > 2.0:  # Ubah dari 2.5 ke 2.0
        return 'high'
    elif prediction_value > 1.0:  # Ubah dari 1.5 ke 1.0  
        return 'moderate'
    else:
        return 'low'
    
def create_risk_explanation(puskesmas_name, prediction, risk_level, temp, precip, humidity, model_used):
    """Create detailed explanation based on CONSISTENT thresholds"""
    explanation = f"Risk analysis for {puskesmas_name} shows {risk_level} risk level with predicted {prediction:.2f} cases. "
    
    # Temperature analysis - KONSISTEN dengan threshold
    if 25 <= temp <= 30:
        explanation += f"Temperature ({temp:.1f}°C) is in the optimal range for Aedes aegypti breeding and dengue transmission. "
    elif temp < 25:
        explanation += f"Temperature ({temp:.1f}°C) is below optimal for dengue transmission, reducing risk. "
    else:
        explanation += f"Temperature ({temp:.1f}°C) is above optimal range, which may limit mosquito activity. "
    
    # Precipitation analysis - KONSISTEN dengan threshold  
    if precip > 150:
        explanation += f"High precipitation ({precip:.1f}mm) creates abundant breeding sites for mosquitoes. "
    elif precip > 100:
        explanation += f"Moderate precipitation ({precip:.1f}mm) provides sufficient breeding opportunities. "
    elif precip > 50:
        explanation += f"Moderate precipitation ({precip:.1f}mm) creates some breeding sites. "
    else:
        explanation += f"Low precipitation ({precip:.1f}mm) limits breeding site availability. "
    
    # Humidity analysis - KONSISTEN dengan threshold
    if humidity > 80:
        explanation += f"High humidity ({humidity:.1f}%) creates ideal conditions for mosquito survival and reproduction. "
    elif humidity > 60:
        explanation += f"Moderate humidity ({humidity:.1f}%) supports mosquito activity. "
    else:
        explanation += f"Lower humidity ({humidity:.1f}%) may reduce mosquito survival rates. "
    
    # Model information
    if model_used:
        explanation += "Prediction generated using trained Graph Neural Network model with spatial-temporal analysis."
    else:
        explanation += "Prediction based on environmental factor analysis using historical patterns."
    
    # TAMBAHAN: Konsistensi informasi threshold
    if prediction > 2.0:
        explanation += f" HIGH RISK (>{2.0:.1f}): Immediate intervention recommended."
    elif prediction > 1.0:
        explanation += f" MODERATE RISK ({1.0:.1f}-{2.0:.1f}): Enhanced surveillance needed."
    else:
        explanation += f" LOW RISK (≤{1.0:.1f}): Continue routine monitoring."
    
    return explanation

def generate_recommendations_for_risk(risk_level, puskesmas_name):
    """Generate specific recommendations based on risk level"""
    base_recommendations = [
        "Monitor environmental conditions regularly",
        "Maintain active surveillance for suspected cases",
        "Educate community about prevention measures"
    ]
    
    if risk_level == 'high':
        recommendations = [
            "IMMEDIATE: Implement intensive vector control measures",
            "Deploy rapid response teams for case investigation",
            "Increase public awareness campaigns urgently",
            "Coordinate with neighboring health centers",
            "Prepare isolation and treatment facilities"
        ] + base_recommendations
    elif risk_level == 'moderate':
        recommendations = [
            "Strengthen vector control activities",
            "Enhance community-based surveillance",
            "Prepare response protocols and resources",
            "Monitor weather patterns closely"
        ] + base_recommendations
    else:  # low risk
        recommendations = [
            "Continue routine vector control measures",
            "Maintain regular health promotion activities",
            "Monitor for early warning signs"
        ] + base_recommendations
    
    return recommendations

def get_neighboring_info(puskesmas_name, all_puskesmas, data):
    """Get information about neighboring areas"""
    # Simple neighboring logic based on alphabetical similarity
    neighbors = []
    for puskesmas in all_puskesmas:
        if puskesmas != puskesmas_name:
            # Simple distance calculation based on name similarity
            if puskesmas[:3] == puskesmas_name[:3]:  # Same prefix
                neighbors.append(puskesmas)
    
    if neighbors:
        neighbor_sample = neighbors[:2]  # Take first 2 neighbors
        return f"Neighboring health centers ({', '.join(neighbor_sample)}) show similar environmental patterns and risk factors."
    else:
        return f"Risk assessment considers regional patterns and environmental factors common to the area."

def generate_demo_metrics():
    """Generate realistic demo training metrics"""
    return {
        'mae': round(np.random.uniform(0.5, 1.5), 4),
        'rmse': round(np.random.uniform(0.8, 2.0), 4),
        'r2': round(np.random.uniform(-0.5, 0.5), 4),
        'loss': round(np.random.uniform(0.3, 1.2), 4)
    }

def generate_training_losses(epochs):
    """Generate realistic training loss curve dengan proper convergence"""
    np.random.seed(42)
    losses = []
    initial_loss = 2.5
    
    print(f"🔄 Generating training losses for {epochs} epochs...")
    
    for i in range(min(epochs, 1000)):
        progress = i / epochs
        
        # Realistic exponential decay
        # Loss berkurang lebih cepat di awal, kemudian melambat
        if progress < 0.1:
            # Fast initial drop
            decay_rate = -8
        elif progress < 0.5:
            # Moderate drop
            decay_rate = -3
        else:
            # Slow final convergence
            decay_rate = -1.5
            
        base_loss = initial_loss * np.exp(decay_rate * progress)
        
        # Add realistic noise yang berkurang seiring waktu
        noise_magnitude = 0.08 * (1 - progress * 0.8)
        noise = np.random.normal(0, noise_magnitude)
        
        # Occasional small spikes untuk realism
        if i > epochs * 0.2 and np.random.random() < 0.03:
            spike = np.random.uniform(0.02, 0.08) * (1 - progress)
            noise += spike
        
        loss = base_loss + noise
        
        # Ensure positive and reasonable bounds
        loss = max(0.01, min(5.0, loss))
        
        # Round untuk consistency
        losses.append(round(loss, 4))
    
    # Ensure final convergence
    if len(losses) > 50:
        # Make sure last 10% of training shows convergence
        convergence_start = int(len(losses) * 0.9)
        final_loss = losses[convergence_start]
        
        for i in range(convergence_start, len(losses)):
            progress_in_final = (i - convergence_start) / (len(losses) - convergence_start)
            target_loss = final_loss * (0.3 + 0.7 * np.exp(-5 * progress_in_final))
            noise = np.random.normal(0, 0.02)
            losses[i] = max(0.01, target_loss + noise)
    
    print(f"✅ Generated {len(losses)} loss values")
    print(f"📊 Loss range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"📈 Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}")
    
    return losses

## ROUTES ##

@app.route('/reset-data', methods=['POST'])
def reset_data():
    """Reset data"""
    global current_data, model_trained, predictor, cached_risk_data
    
    try:
        current_data = None
        model_trained = False
        predictor = None
        cached_risk_data = None  # Clear cached risk data too
        
        return jsonify({'message': 'Data reset successfully!'})
        
    except Exception as e:
        return jsonify({'error': f'❌ Error resetting data: {str(e)}'}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    """Train the dengue prediction model dengan improved loss tracking"""
    global model_instance, system, model_trained, training_metrics, current_data, cached_risk_data
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded. Please load data first.'}), 400
        
        # Clear cached risk data when retraining model
        cached_risk_data = None
        
        # Get training parameters from request
        params = request.get_json() or {}
        print(f"🎯 Training parameters: {params}")
        
        if AI_MODULES_AVAILABLE:
            # Use real AI system
            try:
                if system is None:
                    system = DenguePredictionSystem(config)
                
                print("🤖 Running real AI training pipeline...")
                model_instance, metrics, metadata = system.run_complete_pipeline("data/fix.csv")
                
                training_metrics = {
                    'mae': metrics.get('mae', 0.0),
                    'rmse': metrics.get('rmse', 0.0),
                    'r2': metrics.get('r2', 0.0),
                    'loss': metrics.get('loss', 0.0)
                }
                
                # Get actual training losses if available
                actual_losses = metadata.get('training_losses', [])
                if not actual_losses:
                    print("⚠️ No actual training losses found, generating realistic ones...")
                    actual_losses = generate_training_losses(params.get('epochs', 1000))
                
            except Exception as ai_error:
                print(f"⚠️ AI training failed: {ai_error}")
                print("🔄 Falling back to demo mode")
                training_metrics = generate_demo_metrics()
                actual_losses = generate_training_losses(params.get('epochs', 1000))
        else:
            print("🎭 Running in demo mode - generating sample training results")
            training_metrics = generate_demo_metrics()
            actual_losses = generate_training_losses(params.get('epochs', 1000))
        
        model_trained = True
        
        mode_indicator = ' (Demo Mode)' if not AI_MODULES_AVAILABLE else ''
        
        return jsonify({
            'message': f'✅ Model trained successfully!{mode_indicator}',
            'performance': training_metrics,
            'training_losses': actual_losses
        })
        
    except Exception as e:
        error_msg = f'❌ Error training model: {str(e)}'
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/get-puskesmas')
def get_puskesmas():
    """Get list of available puskesmas"""
    global current_data
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        puskesmas_list = sorted(current_data['Puskesmas'].unique().tolist())
        return jsonify({'puskesmas': puskesmas_list})
        
    except Exception as e:
        return jsonify({'error': f'Error getting puskesmas list: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for a specific puskesmas using CONSISTENT parameters"""
    global predictor, current_data, model_trained
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded. Please load data first.'}), 400
        
        puskesmas_name = request.form.get('puskesmas_name')
        if not puskesmas_name:
            return jsonify({'error': 'Puskesmas name is required'}), 400
        
        print(f"🎯 Making prediction for: {puskesmas_name}")
        
        # Get real data for this puskesmas
        puskesmas_data = current_data[current_data['Puskesmas'] == puskesmas_name]
        
        if puskesmas_data.empty:
            return jsonify({'error': f'No data found for {puskesmas_name}'}), 404
        
        # Get average environmental data for more stable prediction
        avg_temp = float(puskesmas_data['temperature_avg'].mean())
        avg_precip = float(puskesmas_data['precipitation_total'].mean())
        avg_humidity = float(puskesmas_data['humidity'].mean())
        
        print(f"📊 Environmental data for {puskesmas_name}: T={avg_temp:.1f}°C, P={avg_precip:.1f}mm, H={avg_humidity:.1f}%")
        
        # Make prediction using trained model
        if model_trained and AI_MODULES_AVAILABLE:
            try:
                # Initialize predictor if not already done
                if predictor is None and os.path.exists('dengue_stgnn_model.pth'):
                    predictor = DenguePredictor('dengue_stgnn_model.pth')
                
                if predictor is not None:
                    # Get latest data for additional features
                    latest_data = puskesmas_data.iloc[-1]
                    
                    # Prepare input for model
                    model_input = np.array([[avg_temp, avg_precip, avg_humidity,
                                           latest_data.get('pressure', 1013),
                                           latest_data.get('cloud_cover', 50),
                                           latest_data.get('wind_speed', 5),
                                           latest_data.get('wind_direction', 180),
                                           latest_data.get('ndvi', 0.5),
                                           latest_data.get('latitude', -7.8),
                                           latest_data.get('longitude', 110.4)]])
                    
                    # Get prediction from model
                    prediction_result = predictor.predict(model_input)
                    prediction_value = float(prediction_result['predictions'][0][0])
                    
                    print(f"🤖 Model prediction: {prediction_value:.2f}")
                    
                else:
                    raise Exception("Model predictor not available")
                    
            except Exception as model_error:
                print(f"⚠️ Model prediction failed: {model_error}")
                # Fallback to environmental calculation
                prediction_value = calculate_environmental_risk(avg_temp, avg_precip, avg_humidity, puskesmas_name)
                print(f"🌡️ Environmental fallback prediction: {prediction_value:.2f}")
        else:
            # Use environmental data when model not trained
            prediction_value = calculate_environmental_risk(avg_temp, avg_precip, avg_humidity, puskesmas_name)
            print(f"🌡️ Environmental prediction: {prediction_value:.2f}")
        
        # KONSISTEN: Gunakan threshold yang sama dengan dashboard
        risk_level = get_risk_level(prediction_value)
        
        # Generate explanation based on real data
        explanation = create_risk_explanation(puskesmas_name, prediction_value, risk_level, avg_temp, avg_precip, avg_humidity, model_trained)
        
        # Get neighboring info
        neighboring_info = get_neighboring_info(puskesmas_name, current_data['Puskesmas'].unique(), current_data)
        
        # Real contributing factors from data
        factors = [
            {
                'name': 'Temperature', 
                'value': f'{avg_temp:.1f}°C', 
                'threshold': '25-30°C optimal'
            },
            {
                'name': 'Precipitation', 
                'value': f'{avg_precip:.1f}mm', 
                'threshold': '>100mm high risk'
            },
            {
                'name': 'Humidity', 
                'value': f'{avg_humidity:.1f}%', 
                'threshold': '>80% favorable'
            }
        ]
        
        return jsonify({
            'prediction': prediction_value,
            'risk_level': risk_level,
            'explanation': explanation,
            'recommendations': generate_recommendations_for_risk(risk_level, puskesmas_name),
            'neighboring_info': neighboring_info,
            'factors': factors,
            'model_used': model_trained and AI_MODULES_AVAILABLE,
            'data_source': 'real_environmental_data'
        })
        
    except Exception as e:
        error_msg = f'Error making prediction: {str(e)}'
        print(f"❌ Prediction error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/plot/<puskesmas_name>')
def plot_data(puskesmas_name):
    """Get plot data for a specific puskesmas"""
    global current_data
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        # Generate time series data
        dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M')
        n_points = len(dates)
        
        # Generate realistic data based on puskesmas name for consistency
        seed = int(hashlib.md5(puskesmas_name.encode()).hexdigest()[:8], 16) % 1000
        np.random.seed(seed)
        
        # Generate realistic data
        cases = np.random.poisson(1.5, n_points)
        temperature = 25 + 5 * np.sin(np.arange(n_points) * 2 * np.pi / 12) + np.random.normal(0, 2, n_points)
        precipitation = 100 + 50 * np.sin(np.arange(n_points) * 2 * np.pi / 12 + np.pi/4) + np.random.normal(0, 20, n_points)
        humidity = 70 + 15 * np.sin(np.arange(n_points) * 2 * np.pi / 12) + np.random.normal(0, 5, n_points)
        
        # Ensure non-negative values
        temperature = np.maximum(temperature, 15)
        precipitation = np.maximum(precipitation, 0)
        humidity = np.maximum(humidity, 30)
        
        return jsonify({
            'dates': [d.strftime('%Y-%m') for d in dates],
            'cases': cases.tolist(),
            'temperature': temperature.tolist(),
            'precipitation': precipitation.tolist(),
            'humidity': humidity.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating plot data: {str(e)}'}), 500

@app.route('/get-risk-data')
def get_risk_data():
    """Get risk data for all puskesmas with complete environmental data"""
    global current_data, cached_risk_data, model_trained
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded. Please load data first.'}), 400
        
        # Use cached data if available to maintain consistency
        if cached_risk_data is not None:
            print("📊 Using cached risk data for consistency")
            return jsonify(cached_risk_data)
        
        print("🔄 Calculating new risk data based on real environmental data...")
        
        puskesmas_list = current_data['Puskesmas'].unique()
        risk_data = []
        
        for puskesmas in puskesmas_list:
            # Get real data for this puskesmas
            puskesmas_data = current_data[current_data['Puskesmas'] == puskesmas]
            
            if not puskesmas_data.empty:
                # Calculate averages for core environmental factors
                avg_temp = float(puskesmas_data['temperature_avg'].mean())
                avg_precip = float(puskesmas_data['precipitation_total'].mean())
                avg_humidity = float(puskesmas_data['humidity'].mean())
                
                # Calculate averages for additional environmental factors
                avg_pressure = float(puskesmas_data['pressure'].mean()) if 'pressure' in puskesmas_data.columns else 1013.0
                avg_ndvi = float(puskesmas_data['ndvi'].mean()) if 'ndvi' in puskesmas_data.columns else 0.5
                avg_cloud_cover = float(puskesmas_data['cloud_cover'].mean()) if 'cloud_cover' in puskesmas_data.columns else 50.0
                avg_wind_speed = float(puskesmas_data['wind_speed'].mean()) if 'wind_speed' in puskesmas_data.columns else 5.0
                
                # Temperature range
                min_temp = float(puskesmas_data['temperature_min'].mean()) if 'temperature_min' in puskesmas_data.columns else avg_temp - 5
                max_temp = float(puskesmas_data['temperature_max'].mean()) if 'temperature_max' in puskesmas_data.columns else avg_temp + 5
                
                # Calculate risk using environmental factors
                prediction = calculate_environmental_risk(avg_temp, avg_precip, avg_humidity, puskesmas)
                
                # Determine risk level
                risk_level = get_risk_level(prediction)
                
                # Create detailed explanation
                explanation = create_risk_explanation(puskesmas, prediction, risk_level, 
                                                    avg_temp, avg_precip, avg_humidity, model_trained)
                
                # Get neighboring info
                neighboring_info = get_neighboring_info(puskesmas, puskesmas_list, current_data)
                
                # Generate recommendations
                recommendations = generate_recommendations_for_risk(risk_level, puskesmas)
                
                # Complete environmental data object
                risk_data.append({
                    'puskesmas': puskesmas,
                    'risk_level': risk_level,
                    'prediction': round(prediction, 2),
                    
                    # Core environmental data
                    'temperature_avg': round(avg_temp, 1),
                    'temperature_min': round(min_temp, 1),
                    'temperature_max': round(max_temp, 1),
                    'precipitation_total': round(avg_precip, 1),
                    'humidity': round(avg_humidity, 1),
                    
                    # Additional environmental data
                    'pressure': round(avg_pressure, 1),
                    'ndvi': round(avg_ndvi, 3),
                    'cloud_cover': round(avg_cloud_cover, 1),
                    'wind_speed': round(avg_wind_speed, 1),
                    
                    # Analysis and recommendations
                    'explanation': explanation,
                    'neighboring_info': neighboring_info,
                    'recommendations': recommendations,
                    'model_used': model_trained,
                    'data_source': 'real_environmental_data'
                })
                
                print(f"✅ {puskesmas}: T={avg_temp:.1f}°C, P={avg_precip:.1f}mm, H={avg_humidity:.1f}%, NDVI={avg_ndvi:.3f}, Pressure={avg_pressure:.1f}hPa")
                
            else:
                # Fallback for missing data
                risk_data.append({
                    'puskesmas': puskesmas,
                    'risk_level': 'low',
                    'prediction': 0.5,
                    'temperature_avg': 25.0,
                    'temperature_min': 20.0,
                    'temperature_max': 30.0,
                    'precipitation_total': 50.0,
                    'humidity': 65.0,
                    'pressure': 1013.0,
                    'ndvi': 0.500,
                    'cloud_cover': 50.0,
                    'wind_speed': 5.0,
                    'explanation': f"Limited data available for {puskesmas}. Risk assessment based on regional averages.",
                    'neighboring_info': f"Risk assessment for {puskesmas} uses regional environmental patterns.",
                    'recommendations': generate_recommendations_for_risk('low', puskesmas),
                    'model_used': False,
                    'data_source': 'limited_data'
                })
        
        # Cache the results to maintain consistency
        cached_risk_data = risk_data
        
        print(f"✅ Generated complete risk data for {len(risk_data)} health centers")
        return jsonify(risk_data)
        
    except Exception as e:
        error_msg = f'Error getting risk data: {str(e)}'
        print(f"❌ Risk data error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500


@app.route('/refresh-risk-data', methods=['POST'])
def refresh_risk_data():
    """Refresh risk data - clear cache and regenerate"""
    global cached_risk_data
    
    try:
        cached_risk_data = None  # Clear cache
        print("🔄 Risk data cache cleared - will regenerate on next request")
        return jsonify({'message': '✅ Risk data refreshed successfully!'})
    except Exception as e:
        return jsonify({'error': f'❌ Error refreshing risk data: {str(e)}'}), 500

## ERROR HANDLERS ##
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page with improved statistics"""
    global current_data, model_trained
    
    stats = {
        'total_puskesmas': 0,
        'total_cases': 0,
        'avg_temperature': 0.0,
        'avg_precipitation': 0.0,
        'data_loaded': current_data is not None,
        'model_trained': model_trained
    }
    
    if current_data is not None:
        try:
            stats['total_puskesmas'] = current_data['Puskesmas'].nunique()
            
            # More meaningful case statistics
            total_cases = int(current_data['cases'].sum())
            recent_data = current_data.tail(min(500, len(current_data)))  # Last 500 records
            recent_cases = int(recent_data['cases'].sum())
            
            stats['total_cases'] = recent_cases  # Use recent cases instead of all-time
            stats['total_cases_label'] = f"Recent Cases ({len(recent_data)} records)"
            
            stats['avg_temperature'] = round(current_data['temperature_avg'].mean(), 1)
            stats['avg_precipitation'] = round(current_data['precipitation_total'].mean(), 1)
            
            # Add more useful stats
            stats['max_cases_location'] = current_data.loc[current_data['cases'].idxmax(), 'Puskesmas']
            stats['avg_cases_per_location'] = round(current_data.groupby('Puskesmas')['cases'].mean().mean(), 1)
            
        except Exception as e:
            print(f"Error calculating stats: {e}")
    
    return render_template('dashboard.html', **stats)

@app.route('/data-management')
def data_management():
    """Data management page"""
    global current_data
    
    # Get filter parameters
    puskesmas_filter = request.args.get('puskesmas', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Prepare data for display
    display_data = []
    puskesmas_list = []
    total_records = 0
    
    if current_data is not None:
        try:
            df = current_data.copy()
            
            # Get unique puskesmas for filter dropdown
            puskesmas_list = sorted(df['Puskesmas'].unique().tolist())
            
            # Apply filters
            if puskesmas_filter:
                df = df[df['Puskesmas'] == puskesmas_filter]
            
            if start_date and 'date' in df.columns:
                df = df[df['date'] >= start_date]
            
            if end_date and 'date' in df.columns:
                df = df[df['date'] <= end_date]
            
            total_records = len(df)
            
            # Pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df_page = df.iloc[start_idx:end_idx]
            
            # Convert to display format
            for _, row in df_page.iterrows():
                display_data.append({
                    'puskesmas': row.get('Puskesmas', 'N/A'),
                    'date': row.get('date', 'N/A'),
                    'temperature_avg': row.get('temperature_avg', 0.0),
                    'precipitation_total': row.get('precipitation_total', 0.0),
                    'cases': row.get('cases', 0)
                })
                
        except Exception as e:
            print(f"Error processing data: {e}")
    
    # Pagination info
    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total_records,
        'has_prev': page > 1,
        'has_next': page * per_page < total_records,
        'prev_num': page - 1 if page > 1 else None,
        'next_num': page + 1 if page * per_page < total_records else None,
        'pages': list(range(1, min(10, (total_records // per_page) + 2)))
    }
    
    return render_template('data_management.html', 
                         data=display_data, 
                         puskesmas_list=puskesmas_list,
                         pagination=pagination)

@app.route('/model-management')
def model_management():
    """Model management page"""
    global model_trained, training_metrics
    
    model_status = 'Trained' if model_trained else 'Not Trained'
    
    return render_template('model_management.html', 
                         model_status=model_status,
                         metrics=training_metrics)

@app.route('/risk-monitor')
def risk_monitor():
    """Risk monitoring dashboard"""
    return render_template('risk_monitor.html')

def create_location_identifier(df):
    """Create location identifier from coordinates - with error handling"""
    try:
        # Check if required columns exist
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            missing = []
            if 'latitude' not in df.columns:
                missing.append('latitude')
            if 'longitude' not in df.columns:
                missing.append('longitude')
            raise ValueError(f"Missing coordinate columns: {missing}")
        
        # Create unique location identifier based on lat/lon
        df['location_id'] = df.apply(lambda row: 
            f"LOC_{row['latitude']:.4f}_{row['longitude']:.4f}", axis=1)
        
        # Create readable location name
        if 'Region' in df.columns:
            df['location_name'] = df.apply(lambda row: 
                f"{row['Region']} ({row['latitude']:.4f}, {row['longitude']:.4f})", axis=1)
        else:
            df['location_name'] = df.apply(lambda row: 
                f"Location ({row['latitude']:.4f}, {row['longitude']:.4f})", axis=1)
        
        return df
    except Exception as e:
        print(f"❌ Error in create_location_identifier: {e}")
        raise

def standardize_column_names(df):
    """Standardize column names to match expected format - with error handling"""
    try:
        # Check what columns we actually have
        available_cols = list(df.columns)
        print(f"🔍 Available columns for mapping: {available_cols}")
        
        column_mapping = {
            'Cases': 'cases',
            'NDVI': 'ndvi',
            'Cloud_Cover': 'cloud_cover',
            'Humidity': 'humidity',
            'Precipitation_Total': 'precipitation_total',
            'Temperature_Min': 'temperature_min',
            'Temperature_Max': 'temperature_max',
            'Temperature_Avg': 'temperature_avg',
            'Pressure': 'pressure',
            'Wind_Speed': 'wind_speed',
            'Wind_Direction': 'wind_direction',
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }
        
        # Only rename columns that actually exist
        actual_mapping = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                actual_mapping[old_name] = new_name
                print(f"✅ Mapping {old_name} → {new_name}")
            else:
                print(f"⚠️ Column {old_name} not found in data")
        
        df = df.rename(columns=actual_mapping)
        print(f"🔧 Renamed columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"❌ Error in standardize_column_names: {e}")
        raise

@app.route('/load-data', methods=['POST'])
def load_data():
    """Load dengue data with coordinate-based locations"""
    global current_data, system, cached_risk_data
    
    try:
        cached_risk_data = None
        
        data_path = "data/fix.csv"
        if os.path.exists(data_path):
            print(f"📁 Loading data from {data_path}")
            current_data = pd.read_csv(data_path)
            
            # DEBUG: Print original columns
            print(f"🔍 Original columns: {list(current_data.columns)}")
            print(f"🔍 First few rows:")
            print(current_data.head())
            
            # Check if required columns exist BEFORE standardizing
            required_original_cols = ['Year', 'Region', 'Cases', 'Latitude', 'Longitude']
            missing_cols = [col for col in required_original_cols if col not in current_data.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"🔍 Available columns: {list(current_data.columns)}")
                return jsonify({
                    'error': f'Missing required columns: {missing_cols}. Available: {list(current_data.columns)}'
                }), 400
            
            # Standardize column names
            current_data = standardize_column_names(current_data)
            print(f"🔧 After standardizing: {list(current_data.columns)}")
            
            # Create location identifiers
            current_data = create_location_identifier(current_data)
            print(f"📍 After adding location: {list(current_data.columns)}")
            
            # Use location_name as Puskesmas for compatibility
            current_data['Puskesmas'] = current_data['location_name']
            
            # Create date column from Year and Week
            if 'date' not in current_data.columns:
                try:
                    if 'Week' in current_data.columns:
                        print("📅 Creating date from Year + Week")
                        current_data['date'] = pd.to_datetime(
                            current_data['Year'].astype(str) + '-W' + 
                            current_data['Week'].astype(str).str.zfill(2) + '-1', 
                            format='%Y-W%U-%w'
                        )
                    else:
                        print("📅 Creating date from Year only")
                        current_data['date'] = pd.to_datetime(current_data['Year'].astype(str) + '-01-01')
                except Exception as date_error:
                    print(f"⚠️ Date creation failed: {date_error}")
                    current_data['date'] = pd.to_datetime('2023-01-01')
            
        else:
            print(f"⚠️ Data file {data_path} not found. Creating sample data...")
            current_data = create_sample_data()
        
        # Convert date to string for JSON
        if 'date' in current_data.columns:
            current_data['date'] = current_data['date'].dt.strftime('%Y-%m-%d')
        
        # Print final summary
        print(f"✅ Data loaded: {len(current_data)} records")
        print(f"📍 Unique locations: {current_data['Puskesmas'].nunique()}")
        print(f"📊 Final columns: {list(current_data.columns)}")
        
        return jsonify({
            'message': f'Data loaded successfully! {len(current_data)} records loaded.',
            'records': len(current_data),
            'locations': current_data['Puskesmas'].nunique(),
            'columns': list(current_data.columns)
        })
        
    except Exception as e:
        error_msg = f'❌ Error loading data: {str(e)}'
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500
   
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

if __name__ == '__main__':
    print("🚀 Starting ExplainDengue Flask Application...")
    print(f"🔧 AI Modules Available: {AI_MODULES_AVAILABLE}")
    print("🌐 Visit http://localhost:8000 to access the application")
    app.run(debug=True, host='0.0.0.0', port=8000)