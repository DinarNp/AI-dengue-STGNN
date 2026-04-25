"""
Database models for Dengue Prediction Web Application
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model with role-based access control"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'admin', 'health_district', 'public'
    regency = db.Column(db.String(100), nullable=True)  # For health district office users
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    dengue_cases = db.relationship('DengueCase', backref='reporter', lazy='dynamic')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'
    
    def is_health_district(self):
        """Check if user is health district office"""
        return self.role == 'health_district'
    
    def can_edit_regency(self, regency_name):
        """Check if user can edit data for specific regency"""
        if self.is_admin():
            return True
        if self.is_health_district() and self.regency == regency_name:
            return True
        return False
    
    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


class Regency(db.Model):
    """Regency/Kabupaten model"""
    __tablename__ = 'regencies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    population = db.Column(db.Integer)
    area_km2 = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    dengue_cases = db.relationship('DengueCase', backref='regency', lazy='dynamic')
    climate_data = db.relationship('ClimateData', backref='regency', lazy='dynamic')
    ndvi_data = db.relationship('NDVIData', backref='regency', lazy='dynamic')
    predictions = db.relationship('Prediction', backref='regency', lazy='dynamic')
    
    def __repr__(self):
        return f'<Regency {self.name}>'


class DengueCase(db.Model):
    """Dengue case data model"""
    __tablename__ = 'dengue_cases'
    
    id = db.Column(db.Integer, primary_key=True)
    regency_id = db.Column(db.Integer, db.ForeignKey('regencies.id'), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    week = db.Column(db.Integer, nullable=True)  # Epidemiological week
    cases = db.Column(db.Integer, nullable=False, default=0)
    data_source = db.Column(db.String(50), default='manual')  # 'manual', 'skdr', 'dinkes'
    reported_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = db.Column(db.Text)
    
    # Unique constraint for regency-year-month combination
    __table_args__ = (
        db.UniqueConstraint('regency_id', 'year', 'month', name='unique_regency_year_month'),
    )
    
    def __repr__(self):
        return f'<DengueCase {self.regency.name} {self.year}-{self.month}: {self.cases}>'


class ClimateData(db.Model):
    """Climate/weather data model"""
    __tablename__ = 'climate_data'
    
    id = db.Column(db.Integer, primary_key=True)
    regency_id = db.Column(db.Integer, db.ForeignKey('regencies.id'), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    week = db.Column(db.Integer, nullable=True)
    
    # Climate variables
    temperature_min = db.Column(db.Float)
    temperature_max = db.Column(db.Float)
    temperature_avg = db.Column(db.Float)
    humidity = db.Column(db.Float)
    precipitation_total = db.Column(db.Float)
    pressure = db.Column(db.Float)
    wind_speed = db.Column(db.Float)
    wind_direction = db.Column(db.Float)
    cloud_cover = db.Column(db.Float)
    
    # Metadata
    data_source = db.Column(db.String(50))  # 'nasa_power', 'openweather', 'manual'
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (
        db.UniqueConstraint('regency_id', 'year', 'month', name='unique_climate_regency_year_month'),
    )
    
    def __repr__(self):
        return f'<ClimateData {self.regency.name} {self.year}-{self.month}>'


class NDVIData(db.Model):
    """NDVI (Normalized Difference Vegetation Index) data model"""
    __tablename__ = 'ndvi_data'
    
    id = db.Column(db.Integer, primary_key=True)
    regency_id = db.Column(db.Integer, db.ForeignKey('regencies.id'), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    week = db.Column(db.Integer, nullable=True)
    
    ndvi_value = db.Column(db.Float, nullable=False)
    data_source = db.Column(db.String(50), default='modis')  # 'modis', 'landsat', 'manual'
    processing_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_imputed = db.Column(db.Boolean, default=False)  # True if value was filled by imputation
    
    # Unique constraint
    __table_args__ = (
        db.UniqueConstraint('regency_id', 'year', 'month', name='unique_ndvi_regency_year_month'),
    )
    
    def __repr__(self):
        return f'<NDVIData {self.regency.name} {self.year}-{self.month}: {self.ndvi_value}>'


class Prediction(db.Model):
    """Dengue prediction model results"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    regency_id = db.Column(db.Integer, db.ForeignKey('regencies.id'), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    
    predicted_cases = db.Column(db.Float, nullable=False)
    zero_probability = db.Column(db.Float)  # Probability of zero cases
    confidence_lower = db.Column(db.Float)  # Lower confidence bound
    confidence_upper = db.Column(db.Float)  # Upper confidence bound
    
    actual_cases = db.Column(db.Integer)  # Filled in after month ends
    
    model_version = db.Column(db.String(50))
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Risk classification
    risk_level = db.Column(db.String(20))      # 'alert', 'no_alert'
    alert_threshold = db.Column(db.Float)       # mean + 1.25*SD threshold used
    hist_mean = db.Column(db.Float)             # historical mean for that regency+month
    hist_sd = db.Column(db.Float)               # historical SD for that regency+month
    exceed_probability = db.Column(db.Float)    # P(actual > threshold | ~N(predicted, hist_sd)) %
    
    def __repr__(self):
        return f'<Prediction {self.regency.name} {self.year}-{self.month}: {self.predicted_cases:.1f}>'


class ModelVersion(db.Model):
    """Track different model versions"""
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    version_name = db.Column(db.String(100), unique=True, nullable=False)
    model_file = db.Column(db.String(255), nullable=False)  # Path to .pth file
    description = db.Column(db.Text)
    
    # Performance metrics
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    
    # Training info
    training_data_file = db.Column(db.String(255))
    training_date = db.Column(db.DateTime)
    training_samples = db.Column(db.Integer)
    
    is_active = db.Column(db.Boolean, default=False)  # Only one active model at a time
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelVersion {self.version_name}>'


class DataProcessingLog(db.Model):
    """Log of data processing activities"""
    __tablename__ = 'data_processing_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    process_type = db.Column(db.String(50), nullable=False)  # 'climate_fetch', 'ndvi_process', 'manual_input', etc.
    status = db.Column(db.String(20), nullable=False)  # 'started', 'completed', 'failed'
    
    # Details
    records_processed = db.Column(db.Integer)
    error_message = db.Column(db.Text)
    details = db.Column(db.JSON)  # Additional metadata
    
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<DataProcessingLog {self.process_type} - {self.status}>'
