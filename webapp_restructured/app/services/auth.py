"""
Authentication Service
Handles user authentication and authorization
"""
from functools import wraps
from flask import redirect, url_for, flash, session
from flask_login import current_user


def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        if not current_user.is_admin():
            flash('You need admin privileges to access this page.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function


def district_health_required(f):
    """Decorator to require district health office role (or admin)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        if not (current_user.is_admin() or current_user.is_district_health()):
            flash('You need district health office privileges to access this page.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function


def check_regency_access(regency_id):
    """
    Check if current user can access data for a specific regency
    
    Args:
        regency_id: ID of the regency to check
        
    Returns:
        Boolean indicating access permission
    """
    from ..models import Regency
    
    if not current_user.is_authenticated:
        return False
    
    if current_user.is_admin():
        return True
    
    if current_user.is_district_health():
        regency = Regency.query.get(regency_id)
        if regency and regency.name == current_user.regency:
            return True
    
    return False
