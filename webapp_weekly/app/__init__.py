"""
Flask Application Factory
"""
from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate
import os

from .models import db, User
# Renamed from config/ to app_settings/ so the bare "config" package name is
# free for revision_experiments' ML config.config module -- both directories
# were literally named "config", which caused Python's module-name-based
# import cache to silently return whichever one loaded first regardless of
# sys.path order (this webapp's Flask config had a stale FORECAST_HORIZON=1,
# which got returned when the ML code asked for config.config.Config).
from app_settings.config import config

# Initialize Flask extensions
login_manager = LoginManager()
migrate = Migrate()


def create_app(config_name='development'):
    """
    Create and configure Flask application
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Ensure required folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RAW_DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')), exist_ok=True)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    
    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register blueprints
    from .routes.auth import auth
    from .routes.public import public
    from .routes.admin import admin
    from .routes.health_district import health_district
    
    app.register_blueprint(auth)
    app.register_blueprint(public)
    app.register_blueprint(admin)
    app.register_blueprint(health_district)

    from .i18n import register_i18n
    register_i18n(app)
    
    # Create database tables
    with app.app_context():
        db.create_all()
        
        # Initialize default data if needed
        from .utils.init_data import initialize_default_data
        initialize_default_data()
    
    return app
