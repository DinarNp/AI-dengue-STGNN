"""
Authentication Routes
User login, logout, and registration
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user
from urllib.parse import urlparse
from datetime import datetime

from ..models import db, User

auth = Blueprint('auth', __name__, url_prefix='/auth')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('public.dashboard'))

    next_url = request.args.get('next') or request.form.get('next')
    msg      = request.args.get('msg')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account is inactive. Please contact administrator.', 'danger')
                return redirect(url_for('auth.login'))

            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()

            # If a safe next URL was provided, honour it
            if next_url:
                parsed = urlparse(next_url)
                if not parsed.netloc and not parsed.scheme:
                    return redirect(next_url)

            # Default redirect based on role
            if user.is_admin():
                return redirect(url_for('admin.dashboard'))
            elif user.is_health_district():
                return redirect(url_for('health_district.dashboard'))
            else:
                return redirect(url_for('public.dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('auth/login.html', msg=msg, next_url=next_url)


@auth.route('/logout')
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('public.index'))


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration (disabled by default - admin creates users)
    Enable this if you want self-registration
    """
    flash('Registration is disabled. Please contact administrator to create an account.', 'info')
    return redirect(url_for('auth.login'))
    
    # Uncomment below to enable self-registration
    """
    if current_user.is_authenticated:
        return redirect(url_for('public.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('auth.register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('auth.register'))
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('auth.register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('auth.register'))
        
        # Create new user (default role: public)
        new_user = User(
            username=username,
            email=email,
            role='public'
        )
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html')
    """
