"""
Public Routes
Routes accessible to all users (including non-authenticated)
"""
from flask import Blueprint, render_template, request, jsonify
from datetime import datetime
from sqlalchemy import func

from ..models import db, Regency, DengueCase, Prediction, ClimateData, NDVIData
from ..i18n import get_lang, make_t

_FULL_MONTH_KEYS = ['january','february','march','april','may','june',
                    'july','august','september','october','november','december']

public = Blueprint('public', __name__)


@public.route('/')
def index():
    """Landing page"""
    regencies = Regency.query.filter_by(is_active=True).order_by(Regency.id).all()
    return render_template('public/index.html', regencies=regencies)


@public.route('/dashboard')
def dashboard():
    """
    Public dashboard showing dengue conditions for all regencies
    """
    # Get all active regencies
    regencies = Regency.query.filter_by(is_active=True).all()
    
    # Get current month/year
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Collect data for each regency
    regency_data = []
    
    for regency in regencies:
        # Get latest dengue cases
        latest_case = DengueCase.query.filter_by(
            regency_id=regency.id
        ).order_by(DengueCase.year.desc(), DengueCase.month.desc()).first()
        
        # Get latest prediction
        latest_prediction = Prediction.query.filter_by(
            regency_id=regency.id
        ).order_by(Prediction.year.desc(), Prediction.month.desc()).first()
        
        # Get current year total
        year_total = db.session.query(
            func.sum(DengueCase.cases)
        ).filter(
            DengueCase.regency_id == regency.id,
            DengueCase.year == current_year
        ).scalar() or 0
        
        # Get last 6 months trend
        trend = DengueCase.query.filter_by(
            regency_id=regency.id
        ).order_by(DengueCase.year.desc(), DengueCase.month.desc()).limit(6).all()
        
        trend_data = []
        for case in reversed(trend):
            trend_data.append({
                'month': f"{case.year}-{case.month:02d}",
                'cases': case.cases
            })
        
        regency_data.append({
            'regency': regency,
            'latest_case': latest_case,
            'latest_prediction': latest_prediction,
            'year_total': year_total,
            'trend': trend_data
        })
    
    # Get provincial totals
    provincial_total = db.session.query(
        func.sum(DengueCase.cases)
    ).filter(
        DengueCase.year == current_year
    ).scalar() or 0
    
    # Get overall monthly trend (all regencies combined)
    monthly_trend = db.session.query(
        DengueCase.year,
        DengueCase.month,
        func.sum(DengueCase.cases).label('total_cases')
    ).group_by(
        DengueCase.year,
        DengueCase.month
    ).order_by(
        DengueCase.year.desc(),
        DengueCase.month.desc()
    ).limit(12).all()

    overall_trend = []
    for item in reversed(monthly_trend):
        overall_trend.append({
            'month': f"{item.year}-{item.month:02d}",
            'cases': item.total_cases
        })

    # Build 3-month provincial forecast for chart + table
    _lang = get_lang()
    _t    = make_t(_lang)

    # Find latest actual month across all regencies
    latest_any = DengueCase.query.order_by(
        DengueCase.year.desc(), DengueCase.month.desc()
    ).first()
    base_y = latest_any.year  if latest_any else current_year
    base_m = latest_any.month if latest_any else current_month - 1

    provincial_forecast = []  # for chart
    forecast_table = []       # for simple table (per month → per regency)
    py, pm = base_y, base_m
    for _ in range(3):
        pm += 1
        if pm > 12:
            pm = 1; py += 1
        month_label = f"{_t('month.' + _FULL_MONTH_KEYS[pm - 1])} {py}"
        preds = Prediction.query.filter_by(year=py, month=pm).all()
        total_pred = sum(p.predicted_cases for p in preds if p.predicted_cases is not None)
        any_alert  = any(p.risk_level == 'alert' for p in preds)
        provincial_forecast.append({
            'month': f"{py}-{pm:02d}",
            'month_label': month_label,
            'cases': round(total_pred) if preds else None,
            'alert': any_alert,
        })
        # Per-regency rows for table
        rows = []
        pred_map = {p.regency_id: p for p in preds}
        for reg in regencies:
            p = pred_map.get(reg.id)
            rows.append({
                'regency_name': reg.name,
                'predicted': round(p.predicted_cases) if p and p.predicted_cases is not None else None,
                'risk_level': p.risk_level if p else None,
            })
        forecast_table.append({'month_label': month_label, 'rows': rows})

    return render_template('public/dashboard.html',
                         regency_data=regency_data,
                         provincial_total=provincial_total,
                         overall_trend=overall_trend,
                         provincial_forecast=provincial_forecast,
                         forecast_table=forecast_table,
                         current_year=current_year)


@public.route('/regency/<int:regency_id>')
def regency_detail(regency_id):
    """Detailed view for a specific regency"""
    regency = Regency.query.get_or_404(regency_id)
    
    # Get all dengue cases for this regency
    all_cases = DengueCase.query.filter_by(
        regency_id=regency_id
    ).order_by(DengueCase.year.desc(), DengueCase.month.desc()).all()
    
    # Get all predictions
    all_predictions = Prediction.query.filter_by(
        regency_id=regency_id
    ).order_by(Prediction.year.desc(), Prediction.month.desc()).all()
    
    # Get yearly statistics
    yearly_stats = db.session.query(
        DengueCase.year,
        func.sum(DengueCase.cases).label('total_cases'),
        func.avg(DengueCase.cases).label('avg_cases'),
        func.max(DengueCase.cases).label('max_cases')
    ).filter(
        DengueCase.regency_id == regency_id
    ).group_by(
        DengueCase.year
    ).order_by(
        DengueCase.year.desc()
    ).all()
    
    # Get monthly trend for visualization
    monthly_trend = []
    for case in reversed(all_cases[-24:]):  # Last 24 months
        monthly_trend.append({
            'month': f"{case.year}-{case.month:02d}",
            'cases': case.cases
        })
    
    return render_template('public/regency_detail.html',
                         regency=regency,
                         all_cases=all_cases,
                         all_predictions=all_predictions,
                         yearly_stats=yearly_stats,
                         monthly_trend=monthly_trend)


@public.route('/statistics')
def statistics():
    """Provincial-level statistics and analysis"""
    current_year = datetime.now().year
    
    # Get yearly comparison
    years = range(current_year - 4, current_year + 1)
    yearly_comparison = []
    
    for year in years:
        total = db.session.query(
            func.sum(DengueCase.cases)
        ).filter(
            DengueCase.year == year
        ).scalar() or 0
        
        yearly_comparison.append({
            'year': year,
            'total_cases': total
        })
    
    # Get monthly pattern (average cases per month across all years)
    monthly_pattern = db.session.query(
        DengueCase.month,
        func.avg(DengueCase.cases).label('avg_cases')
    ).group_by(
        DengueCase.month
    ).order_by(
        DengueCase.month
    ).all()
    
    _t = make_t(get_lang())
    _short_keys = ['jan','feb','mar','apr','may_short','jun','jul','aug','sep','oct','nov','dec']
    _full_keys  = ['january','february','march','april','may','june',
                   'july','august','september','october','november','december']

    monthly_avg = []
    for item in monthly_pattern:
        monthly_avg.append({
            'month': _t('month.' + _short_keys[item.month - 1]),
            'avg_cases': round(item.avg_cases, 1)
        })

    # Get regency comparison for current year
    regency_comparison = db.session.query(
        Regency.name,
        func.sum(DengueCase.cases).label('total_cases')
    ).join(
        DengueCase, DengueCase.regency_id == Regency.id
    ).filter(
        DengueCase.year == current_year
    ).group_by(
        Regency.name
    ).order_by(
        func.sum(DengueCase.cases).desc()
    ).all()

    regency_totals = []
    for item in regency_comparison:
        # Get latest prediction risk_level for this regency
        reg_obj = Regency.query.filter_by(name=item.name).first()
        latest_pred = None
        if reg_obj:
            latest_pred = Prediction.query.filter_by(regency_id=reg_obj.id)\
                .order_by(Prediction.year.desc(), Prediction.month.desc()).first()
        regency_totals.append({
            'regency': item.name,
            'total_cases': item.total_cases,
            'risk_level': latest_pred.risk_level if latest_pred else None,
        })

    # Calculate key metrics
    total_cases_this_year = sum(item['total_cases'] for item in regency_totals)

    # Get highest risk month (current year)
    highest_month = db.session.query(
        DengueCase.month,
        func.sum(DengueCase.cases).label('total_cases')
    ).filter(
        DengueCase.year == current_year
    ).group_by(
        DengueCase.month
    ).order_by(
        func.sum(DengueCase.cases).desc()
    ).first()

    highest_risk_month = _t('month.' + _full_keys[highest_month.month - 1]) if highest_month else 'N/A'
    
    return render_template('public/statistics.html',
                         yearly_comparison=yearly_comparison,
                         monthly_avg=monthly_avg,
                         regency_totals=regency_totals,
                         total_cases_this_year=total_cases_this_year,
                         highest_risk_month=highest_risk_month,
                         current_year=current_year)


@public.route('/about')
def about():
    """About page - information about the system"""
    return render_template('public/about.html')


@public.route('/api/regencies')
def api_regencies():
    """API endpoint: Get all regencies"""
    regencies = Regency.query.filter_by(is_active=True).all()
    
    return jsonify([{
        'id': r.id,
        'name': r.name,
        'latitude': r.latitude,
        'longitude': r.longitude,
        'population': r.population,
        'area_km2': r.area_km2
    } for r in regencies])


@public.route('/api/cases/<int:regency_id>')
def api_cases(regency_id):
    """API endpoint: Get dengue cases for a regency"""
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    
    query = DengueCase.query.filter_by(regency_id=regency_id)
    
    if year:
        query = query.filter_by(year=year)
    if month:
        query = query.filter_by(month=month)
    
    cases = query.order_by(DengueCase.year.desc(), DengueCase.month.desc()).all()
    
    return jsonify([{
        'year': c.year,
        'month': c.month,
        'cases': c.cases,
        'date': f"{c.year}-{c.month:02d}"
    } for c in cases])


@public.route('/api/predictions/<int:regency_id>')
def api_predictions(regency_id):
    """API endpoint: Get predictions for a regency"""
    predictions = Prediction.query.filter_by(
        regency_id=regency_id
    ).order_by(Prediction.year.desc(), Prediction.month.desc()).all()
    
    return jsonify([{
        'year': p.year,
        'month': p.month,
        'predicted_cases': p.predicted_cases,
        'risk_level': p.risk_level,
        'confidence_lower': p.confidence_lower,
        'confidence_upper': p.confidence_upper,
        'date': f"{p.year}-{p.month:02d}"
    } for p in predictions])
