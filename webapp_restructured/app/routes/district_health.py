"""
District Health Office Routes
Routes for district health office users
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import current_user
from datetime import datetime

from ..models import db, Regency, DengueCase, Prediction
from ..services.auth import district_health_required, check_regency_access
from ..services.data_pipeline import DataPipelineService
from ..services.prediction import PredictionService

district_health = Blueprint('district_health', __name__, url_prefix='/district')


@district_health.route('/dashboard')
@district_health_required
def dashboard():
    """District health office dashboard"""
    # Get user's regency
    regency = Regency.query.filter_by(name=current_user.regency).first()
    
    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get recent dengue cases for this regency
    recent_cases = DengueCase.query.filter_by(
        regency_id=regency.id
    ).order_by(DengueCase.year.desc(), DengueCase.month.desc()).limit(12).all()
    
    # Get latest prediction
    latest_prediction = Prediction.query.filter_by(
        regency_id=regency.id
    ).order_by(Prediction.year.desc(), Prediction.month.desc()).first()
    
    # Calculate statistics
    total_cases_this_year = db.session.query(
        db.func.sum(DengueCase.cases)
    ).filter(
        DengueCase.regency_id == regency.id,
        DengueCase.year == datetime.now().year
    ).scalar() or 0
    
    # Get monthly trend for chart
    monthly_trend = []
    for case in reversed(recent_cases):
        monthly_trend.append({
            'month': f"{case.year}-{case.month:02d}",
            'cases': case.cases
        })
    
    return render_template('district_health/dashboard.html',
                         regency=regency,
                         recent_cases=recent_cases,
                         latest_prediction=latest_prediction,
                         total_cases_this_year=total_cases_this_year,
                         monthly_trend=monthly_trend)


@district_health.route('/update-cases')
@district_health_required
def update_cases():
    """Form to update dengue cases"""
    regency = Regency.query.filter_by(name=current_user.regency).first()
    
    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get existing cases for this year
    current_year = datetime.now().year
    existing_cases = DengueCase.query.filter_by(
        regency_id=regency.id,
        year=current_year
    ).order_by(DengueCase.month).all()
    
    return render_template('district_health/update_cases.html',
                         regency=regency,
                         existing_cases=existing_cases,
                         current_year=current_year)


@district_health.route('/cases/add', methods=['POST'])
@district_health_required
def add_cases():
    """Add or update dengue cases"""
    try:
        data = request.get_json()
        
        # Get user's regency
        regency = Regency.query.filter_by(name=current_user.regency).first()
        
        if not regency:
            return jsonify({'success': False, 'message': 'Invalid regency assignment'}), 403
        
        # Verify user can only update their own regency
        if not check_regency_access(regency.id):
            return jsonify({'success': False, 'message': 'Access denied'}), 403
        
        pipeline = DataPipelineService(request.app.config)
        result = pipeline.add_dengue_cases(
            regency_id=regency.id,
            year=int(data['year']),
            month=int(data['month']),
            cases=int(data['cases']),
            user_id=current_user.id,
            notes=data.get('notes', '')
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@district_health.route('/predictions')
@district_health_required
def view_predictions():
    """View predictions for user's regency"""
    regency = Regency.query.filter_by(name=current_user.regency).first()
    
    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get all predictions for this regency
    predictions = Prediction.query.filter_by(
        regency_id=regency.id
    ).order_by(Prediction.year.desc(), Prediction.month.desc()).limit(12).all()
    
    # Get prediction service for recommendations
    prediction_service = PredictionService(request.app.config)
    
    # Attach recommendations to predictions
    predictions_with_recommendations = []
    for pred in predictions:
        recommendation = prediction_service.get_recommendation(
            pred.predicted_cases,
            pred.risk_level
        )
        predictions_with_recommendations.append({
            'prediction': pred,
            'recommendation': recommendation
        })
    
    return render_template('district_health/predictions.html',
                         regency=regency,
                         predictions=predictions_with_recommendations)


@district_health.route('/request-prediction', methods=['POST'])
@district_health_required
def request_prediction():
    """Request new prediction for upcoming month"""
    try:
        data = request.get_json()
        
        # Get user's regency
        regency = Regency.query.filter_by(name=current_user.regency).first()
        
        if not regency:
            return jsonify({'success': False, 'message': 'Invalid regency assignment'}), 403
        
        year = int(data['year'])
        month = int(data['month'])
        
        # Generate prediction
        prediction_service = PredictionService(request.app.config)
        result = prediction_service.predict_single_regency(regency.id, year, month)
        
        if result['success']:
            # Get recommendation
            recommendation = prediction_service.get_recommendation(
                result['predicted_cases'],
                result['risk_level']
            )
            result['recommendation'] = recommendation
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@district_health.route('/reports')
@district_health_required
def view_reports():
    """View reports and statistics"""
    regency = Regency.query.filter_by(name=current_user.regency).first()
    
    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get yearly statistics
    current_year = datetime.now().year
    years = range(current_year - 3, current_year + 1)
    
    yearly_stats = []
    for year in years:
        total_cases = db.session.query(
            db.func.sum(DengueCase.cases)
        ).filter(
            DengueCase.regency_id == regency.id,
            DengueCase.year == year
        ).scalar() or 0
        
        yearly_stats.append({
            'year': year,
            'total_cases': total_cases
        })
    
    # Get monthly breakdown for current year
    monthly_breakdown = DengueCase.query.filter_by(
        regency_id=regency.id,
        year=current_year
    ).order_by(DengueCase.month).all()
    
    return render_template('district_health/reports.html',
                         regency=regency,
                         yearly_stats=yearly_stats,
                         monthly_breakdown=monthly_breakdown,
                         current_year=current_year)
