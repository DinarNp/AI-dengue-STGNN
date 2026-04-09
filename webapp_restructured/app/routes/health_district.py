"""
District Health Office Routes
Routes for district health office users
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import current_user
from datetime import datetime

from ..models import db, Regency, DengueCase, Prediction
from ..services.auth import health_district_required, check_regency_access
from ..services.data_pipeline import DataPipelineService
from ..services.prediction import PredictionService

health_district = Blueprint('health_district', __name__, url_prefix='/district')


@health_district.route('/dashboard')
@health_district_required
def dashboard():
    """Health district office dashboard"""
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
    
    return render_template('health_district/dashboard.html',
                         regency=regency,
                         recent_cases=recent_cases,
                         latest_prediction=latest_prediction,
                         total_cases_this_year=total_cases_this_year,
                         monthly_trend=monthly_trend)


@health_district.route('/update-cases')
@health_district_required
def update_cases():
    """View, add, and edit dengue cases for user's regency"""
    regency = Regency.query.filter_by(name=current_user.regency).first()

    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))

    current_year = datetime.now().year
    selected_year = request.args.get('year', current_year, type=int)

    # Cases for selected year
    cases = DengueCase.query.filter_by(
        regency_id=regency.id,
        year=selected_year
    ).order_by(DengueCase.month).all()

    # All years that have data (plus current year always available)
    year_rows = (
        db.session.query(DengueCase.year)
        .filter_by(regency_id=regency.id)
        .distinct()
        .order_by(DengueCase.year.desc())
        .all()
    )
    available_years = [y[0] for y in year_rows]
    if current_year not in available_years:
        available_years.insert(0, current_year)

    # Summary stats
    total_selected_year = sum(c.cases for c in cases)
    total_all_time = db.session.query(
        db.func.sum(DengueCase.cases)
    ).filter_by(regency_id=regency.id).scalar() or 0

    return render_template('health_district/update_cases.html',
                           regency=regency,
                           cases=cases,
                           selected_year=selected_year,
                           available_years=available_years,
                           total_selected_year=total_selected_year,
                           total_all_time=total_all_time,
                           now=datetime.now())


@health_district.route('/cases/add', methods=['POST'])
@health_district_required
def add_cases():
    """Add or update dengue cases — direct DB write, no pipeline dependency"""
    try:
        data = request.get_json()
        regency = Regency.query.filter_by(name=current_user.regency).first()

        if not regency:
            return jsonify({'success': False, 'message': 'Invalid regency assignment'}), 403

        year  = int(data['year'])
        month = int(data['month'])
        cases_count = int(data['cases'])
        notes = data.get('notes', '')

        existing = DengueCase.query.filter_by(
            regency_id=regency.id, year=year, month=month
        ).first()

        if existing:
            existing.cases = cases_count
            existing.notes = notes
            existing.updated_at = datetime.utcnow()
            existing.reported_by_id = current_user.id
            action = 'updated'
        else:
            db.session.add(DengueCase(
                regency_id=regency.id,
                year=year,
                month=month,
                cases=cases_count,
                notes=notes,
                data_source='manual',
                reported_by_id=current_user.id
            ))
            action = 'added'

        db.session.commit()
        return jsonify({'success': True, 'action': action,
                        'message': f'Dengue cases {action} successfully'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@health_district.route('/cases/<int:case_id>/edit', methods=['PATCH'])
@health_district_required
def edit_case(case_id):
    """Edit an existing dengue case — only allowed for user's own regency"""
    try:
        case = DengueCase.query.get_or_404(case_id)
        regency = Regency.query.filter_by(name=current_user.regency).first()

        if not regency or case.regency_id != regency.id:
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        data = request.get_json()
        case.cases = int(data['cases'])
        case.updated_at = datetime.utcnow()
        case.reported_by_id = current_user.id
        db.session.commit()

        return jsonify({'success': True, 'cases': case.cases})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@health_district.route('/predictions')
@health_district_required
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
    from flask import current_app as _ca; prediction_service = PredictionService(_ca.config)
    
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
    
    return render_template('health_district/predictions.html',
                         regency=regency,
                         predictions=predictions_with_recommendations)


@health_district.route('/request-prediction', methods=['POST'])
@health_district_required
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
        from flask import current_app as _ca; prediction_service = PredictionService(_ca.config)
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


@health_district.route('/reports')
@health_district_required
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
    
    return render_template('health_district/reports.html',
                         regency=regency,
                         yearly_stats=yearly_stats,
                         monthly_breakdown=monthly_breakdown,
                         current_year=current_year)
