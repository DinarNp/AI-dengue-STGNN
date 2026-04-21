"""
Configuration settings for Dengue Prediction Web Application
"""
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    
    # Basic Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'dengue_app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session config
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False  # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # File upload config
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'tif', 'tiff'}
    
    # Data folders
    RAW_DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    PROCESSED_DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    
    # Model config
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    ACTIVE_MODEL_PATH = os.path.join(MODEL_FOLDER, 'active_model.pth')
    
    # API Keys for climate data
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY') or '7e6ea782527bd138dc36dd188ac36f8f'
    NASA_POWER_API_URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'
    
    # Pagination
    ITEMS_PER_PAGE = 50
    
    # Regencies configuration (Yogyakarta)
    REGENCIES = [
        {
            'name': 'KAB BANTUL',
            'latitude': -7.902328,
            'longitude': 110.2862991,
            'population': 1053971,
            'area_km2': 506.85
        },
        {
            'name': 'KAB GUNUNG KIDUL',
            'latitude': -7.9930685,
            'longitude': 110.2704459,
            'population': 747893,
            'area_km2': 1485.36
        },
        {
            'name': 'KAB KULON PROGO',
            'latitude': -7.8186375,
            'longitude': 110.1613062,
            'population': 436395,
            'area_km2': 586.27
        },
        {
            'name': 'KAB SLEMAN',
            'latitude': -7.7145614,
            'longitude': 110.3544861,
            'population': 1197128,
            'area_km2': 574.82
        },
        {
            'name': 'KOTA YOGYAKARTA',
            'latitude': -7.7955798,
            'longitude': 110.3694896,
            'population': 422732,
            'area_km2': 32.5
        }
    ]
    
    # Risk level thresholds (cases per month)
    RISK_THRESHOLDS = {
        'low': 30,
        'medium': 60,
        'high': 100
    }
    
    # Data processing settings
    API_DELAY_SECONDS = 1.5  # Delay between API calls
    CACHE_EXPIRY_DAYS = 30
    CHECKPOINT_INTERVAL = 10
    
    # NDVI settings
    NDVI_SOURCE = 'MODIS'  # Default NDVI source
    NDVI_PRODUCT = 'MOD_NDVI_16'  # MODIS product
    
    # Feature columns for model
    FEATURE_COLUMNS = [
        'NDVI', 'Cloud_Cover', 'Humidity', 'Precipitation_Total',
        'Temperature_Min', 'Temperature_Max', 'Temperature_Avg',
        'Pressure', 'Wind_Speed', 'Wind_Direction'
    ]
    
    # Model hyperparameters (from original config)
    WINDOW_SIZE_MONTHLY = 4  # 4 months history
    FORECAST_HORIZON = 1  # Predict 1 month ahead
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    NUM_HEADS = 4
    DROPOUT = 0.15
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 16
    EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 100


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True  # Require HTTPS
    
    # Use environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
