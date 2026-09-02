"""
Public Routes
Routes accessible to all users (including non-authenticated)
"""
from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, date
from sqlalchemy import func
from epiweeks import Week

from ..models import db, Regency, DengueCase, Prediction, ClimateData, NDVIData
from ..i18n import get_lang, make_t

public = Blueprint('public', __name__)

# Trend charts and forecast cards are scoped to a fixed year rather than a
# rolling window: actual/reported data covers the year up to the last
# FORECAST_WEEKS_COUNT weeks, and the forecast covers those final weeks.
TARGET_YEAR = 2025
FORECAST_WEEKS_COUNT = 4
_WEEKS_IN_TARGET_YEAR = Week.fromdate(date(TARGET_YEAR, 12, 28), system='iso').week
FORECAST_START_WEEK = _WEEKS_IN_TARGET_YEAR - FORECAST_WEEKS_COUNT + 1
ACTUAL_END_WEEK = FORECAST_START_WEEK - 1


@public.route('/')
def index():
    """Landing page"""
    regencies = Regency.query.filter_by(is_active=True).order_by(Regency.id).all()
    return render_template('public/index.html', regencies=regencies)


@public.route('/dashboard')
def dashboard():
    """
    Public dashboard showing dengue conditions for all regencies (weekly cadence).
    Trend charts are scoped to TARGET_YEAR: actual/reported data covers weeks
    1-ACTUAL_END_WEEK, and the forecast/prediction segment covers the final
    FORECAST_WEEKS_COUNT weeks of the year (FORECAST_START_WEEK-52/53).
    """
    import numpy as _np

    # Get all active regencies
    regencies = Regency.query.filter_by(is_active=True).all()

    _lang = get_lang()
    _t    = make_t(_lang)

    def _historical_threshold(regency_id, epi_week, upto_year):
        hist_vals = db.session.query(DengueCase.cases).filter(
            DengueCase.regency_id == regency_id,
            DengueCase.epi_week == epi_week,
            DengueCase.epi_year >= 2021,
            DengueCase.epi_year <= upto_year - 1
        ).all()
        vals = [r.cases for r in hist_vals if r.cases is not None]
        if len(vals) >= 2:
            hm = float(_np.mean(vals)); hs = float(_np.std(vals, ddof=1))
            return round(hm + 1.25 * hs, 1)
        elif vals:
            return round(float(vals[0]), 1)
        return None

    # Collect data for each regency
    regency_data = []

    for regency in regencies:
        # Get latest dengue cases / prediction (for the stat card, independent of TARGET_YEAR)
        latest_case = DengueCase.query.filter_by(
            regency_id=regency.id
        ).order_by(DengueCase.epi_year.desc(), DengueCase.epi_week.desc()).first()

        latest_prediction = Prediction.query.filter_by(
            regency_id=regency.id
        ).order_by(Prediction.epi_year.desc(), Prediction.epi_week.desc()).first()

        year_total = db.session.query(
            func.sum(DengueCase.cases)
        ).filter(
            DengueCase.regency_id == regency.id,
            DengueCase.epi_year == TARGET_YEAR
        ).scalar() or 0

        # Actual cases for TARGET_YEAR, weeks 1..ACTUAL_END_WEEK
        actual_cases = DengueCase.query.filter(
            DengueCase.regency_id == regency.id,
            DengueCase.epi_year == TARGET_YEAR,
            DengueCase.epi_week <= ACTUAL_END_WEEK
        ).order_by(DengueCase.epi_week).all()
        trend_data = [{'week': f"{TARGET_YEAR}-W{c.epi_week:02d}", 'cases': c.cases} for c in actual_cases]

        # Forecast: last FORECAST_WEEKS_COUNT weeks of TARGET_YEAR
        forecast_weeks = []
        prediction_trend = []
        risk_counts = {'alert': 0, 'no_alert': 0}
        for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
            pred = Prediction.query.filter_by(regency_id=regency.id, epi_year=TARGET_YEAR, epi_week=w).first()
            forecast_weeks.append({'epi_year': TARGET_YEAR, 'epi_week': w, 'prediction': pred})
            prediction_trend.append({
                'week': f"{TARGET_YEAR}-W{w:02d}",
                'cases': round(float(pred.predicted_cases), 1) if pred else None,
            })
            if pred:
                rl = pred.risk_level
                risk_counts[rl] = risk_counts.get(rl, 0) + 1

        risk_level = 'alert' if risk_counts.get('alert', 0) > 0 else \
                     ('no_alert' if risk_counts.get('no_alert', 0) > 0 else None)

        # Threshold line across the actual + forecast weeks
        threshold_trend = [_historical_threshold(regency.id, c.epi_week, TARGET_YEAR) for c in actual_cases]
        for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
            pred2 = Prediction.query.filter_by(regency_id=regency.id, epi_year=TARGET_YEAR, epi_week=w).first()
            threshold_trend.append(round(float(pred2.alert_threshold), 1) if pred2 and pred2.alert_threshold is not None else None)

        regency_data.append({
            'regency': regency,
            'latest_case': latest_case,
            'latest_prediction': latest_prediction,
            'year_total': year_total,
            'trend': trend_data,
            'prediction_trend': prediction_trend,
            'threshold_trend': threshold_trend,
            'forecast_weeks': forecast_weeks,
            'risk_counts': risk_counts,
            'risk_level': risk_level,
        })

    # Get provincial totals for TARGET_YEAR
    provincial_total = db.session.query(
        func.sum(DengueCase.cases)
    ).filter(
        DengueCase.epi_year == TARGET_YEAR
    ).scalar() or 0

    # Overall weekly trend (all regencies combined), TARGET_YEAR weeks 1..ACTUAL_END_WEEK
    weekly_trend_rows = db.session.query(
        DengueCase.epi_week,
        func.sum(DengueCase.cases).label('total_cases')
    ).filter(
        DengueCase.epi_year == TARGET_YEAR,
        DengueCase.epi_week <= ACTUAL_END_WEEK
    ).group_by(
        DengueCase.epi_week
    ).order_by(
        DengueCase.epi_week
    ).all()

    overall_trend = [
        {'week': f"{TARGET_YEAR}-W{item.epi_week:02d}", 'cases': item.total_cases}
        for item in weekly_trend_rows
    ]

    # Provincial threshold — same-epi-week provincial totals from 2021 up to TARGET_YEAR-1
    overall_threshold = []
    for item in overall_trend:
        woi = int(item['week'].split('-W')[1])
        prov_hist = db.session.query(
            func.sum(DengueCase.cases)
        ).filter(
            DengueCase.epi_week == woi,
            DengueCase.epi_year >= 2021,
            DengueCase.epi_year <= TARGET_YEAR - 1
        ).group_by(DengueCase.epi_year).all()
        vals = [float(r[0]) for r in prov_hist if r[0] is not None]
        if len(vals) >= 2:
            hm = float(_np.mean(vals)); hs = float(_np.std(vals, ddof=1))
            overall_threshold.append(round(hm + 1.25 * hs, 1))
        elif vals:
            overall_threshold.append(round(vals[0], 1))
        else:
            overall_threshold.append(None)

    # Provincial forecast for chart + table: last FORECAST_WEEKS_COUNT weeks of TARGET_YEAR
    provincial_forecast = []  # for chart
    forecast_table = []       # for table (per week → per regency)
    for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
        week_label = f"{TARGET_YEAR}-W{w:02d}"
        preds = Prediction.query.filter_by(epi_year=TARGET_YEAR, epi_week=w).all()
        total_pred = sum(p.predicted_cases for p in preds if p.predicted_cases is not None)
        any_alert  = any(p.risk_level == 'alert' for p in preds)
        provincial_forecast.append({
            'week': week_label,
            'week_label': week_label,
            'cases': round(total_pred) if preds else None,
            'alert': any_alert,
        })
        total_threshold = sum(
            p.alert_threshold for p in preds if p.alert_threshold is not None
        )
        overall_threshold.append(round(total_threshold, 1) if preds else None)
        rows = []
        pred_map = {p.regency_id: p for p in preds}
        for reg in regencies:
            p = pred_map.get(reg.id)
            rows.append({
                'regency_name': reg.name,
                'predicted': round(p.predicted_cases) if p and p.predicted_cases is not None else None,
                'risk_level': p.risk_level if p else None,
            })
        forecast_table.append({'week_label': week_label, 'rows': rows})

    # Per-regency chart data (TARGET_YEAR actual + forecast + threshold) for selector chart
    all_chart_data = []
    for reg in regencies:
        actual_cases_reg = DengueCase.query.filter(
            DengueCase.regency_id == reg.id,
            DengueCase.epi_year == TARGET_YEAR,
            DengueCase.epi_week <= ACTUAL_END_WEEK
        ).order_by(DengueCase.epi_week).all()
        weekly_trend_reg = [{'week': f"{TARGET_YEAR}-W{c.epi_week:02d}", 'cases': c.cases} for c in actual_cases_reg]

        pred_trend_reg = []
        for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
            pred = Prediction.query.filter_by(regency_id=reg.id, epi_year=TARGET_YEAR, epi_week=w).first()
            pred_trend_reg.append({
                'week': f"{TARGET_YEAR}-W{w:02d}",
                'cases': round(float(pred.predicted_cases), 1) if pred else None,
            })

        thresh_reg = [_historical_threshold(reg.id, c.epi_week, TARGET_YEAR) for c in actual_cases_reg]
        for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
            pred2 = Prediction.query.filter_by(regency_id=reg.id, epi_year=TARGET_YEAR, epi_week=w).first()
            thresh_reg.append(round(float(pred2.alert_threshold), 1) if pred2 and pred2.alert_threshold is not None else None)

        all_chart_data.append({
            'regency_id': reg.id,
            'regency_name': reg.name,
            'weekly_trend': weekly_trend_reg,
            'prediction_trend': pred_trend_reg,
            'threshold_trend': thresh_reg,
        })

    return render_template('public/dashboard.html',
                         regency_data=regency_data,
                         all_chart_data=all_chart_data,
                         provincial_total=provincial_total,
                         overall_trend=overall_trend,
                         overall_threshold=overall_threshold,
                         provincial_forecast=provincial_forecast,
                         forecast_table=forecast_table,
                         current_year=TARGET_YEAR)


@public.route('/regency/<int:regency_id>')
def regency_detail(regency_id):
    """Detailed view for a specific regency (weekly cadence)"""
    regency = Regency.query.get_or_404(regency_id)

    # Get all dengue cases for this regency
    all_cases = DengueCase.query.filter_by(
        regency_id=regency_id
    ).order_by(DengueCase.epi_year.desc(), DengueCase.epi_week.desc()).all()

    # Get all predictions
    all_predictions = Prediction.query.filter_by(
        regency_id=regency_id
    ).order_by(Prediction.epi_year.desc(), Prediction.epi_week.desc()).all()

    # Get yearly statistics
    yearly_stats = db.session.query(
        DengueCase.epi_year,
        func.sum(DengueCase.cases).label('total_cases'),
        func.avg(DengueCase.cases).label('avg_cases'),
        func.max(DengueCase.cases).label('max_cases')
    ).filter(
        DengueCase.regency_id == regency_id
    ).group_by(
        DengueCase.epi_year
    ).order_by(
        DengueCase.epi_year.desc()
    ).all()

    # Get weekly trend for visualization (TARGET_YEAR, weeks 1..ACTUAL_END_WEEK)
    trend_cases = DengueCase.query.filter(
        DengueCase.regency_id == regency_id,
        DengueCase.epi_year == TARGET_YEAR,
        DengueCase.epi_week <= ACTUAL_END_WEEK
    ).order_by(DengueCase.epi_week).all()
    weekly_trend = [
        {'week': f"{TARGET_YEAR}-W{case.epi_week:02d}", 'cases': case.cases}
        for case in trend_cases
    ]

    return render_template('public/regency_detail.html',
                         regency=regency,
                         all_cases=all_cases,
                         all_predictions=all_predictions,
                         yearly_stats=yearly_stats,
                         weekly_trend=weekly_trend)


@public.route('/statistics')
def statistics():
    """Provincial-level statistics and analysis (weekly cadence)"""
    current_epi_year = Week.thisweek(system='iso').year

    # Get yearly comparison
    years = range(current_epi_year - 4, current_epi_year + 1)
    yearly_comparison = []

    for year in years:
        total = db.session.query(
            func.sum(DengueCase.cases)
        ).filter(
            DengueCase.epi_year == year
        ).scalar() or 0

        yearly_comparison.append({
            'year': year,
            'total_cases': total
        })

    # Get seasonal pattern (average cases per epi-week across all years)
    weekly_pattern = db.session.query(
        DengueCase.epi_week,
        func.avg(DengueCase.cases).label('avg_cases')
    ).group_by(
        DengueCase.epi_week
    ).order_by(
        DengueCase.epi_week
    ).all()

    weekly_avg = [
        {'week': f"W{item.epi_week:02d}", 'avg_cases': round(item.avg_cases, 1)}
        for item in weekly_pattern
    ]

    _t = make_t(get_lang())

    # Get regency comparison for current epi-year
    regency_comparison = db.session.query(
        Regency.name,
        func.sum(DengueCase.cases).label('total_cases')
    ).join(
        DengueCase, DengueCase.regency_id == Regency.id
    ).filter(
        DengueCase.epi_year == current_epi_year
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
                .order_by(Prediction.epi_year.desc(), Prediction.epi_week.desc()).first()
        regency_totals.append({
            'regency': item.name,
            'total_cases': item.total_cases,
            'risk_level': latest_pred.risk_level if latest_pred else None,
        })

    # Calculate key metrics
    total_cases_this_year = sum(item['total_cases'] for item in regency_totals)

    # Get highest-burden epi-week (current epi-year)
    highest_week = db.session.query(
        DengueCase.epi_week,
        func.sum(DengueCase.cases).label('total_cases')
    ).filter(
        DengueCase.epi_year == current_epi_year
    ).group_by(
        DengueCase.epi_week
    ).order_by(
        func.sum(DengueCase.cases).desc()
    ).first()

    highest_risk_week = f"W{highest_week.epi_week:02d}" if highest_week else 'N/A'

    return render_template('public/statistics.html',
                         yearly_comparison=yearly_comparison,
                         weekly_avg=weekly_avg,
                         regency_totals=regency_totals,
                         total_cases_this_year=total_cases_this_year,
                         highest_risk_week=highest_risk_week,
                         current_year=current_epi_year)


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
    epi_year = request.args.get('epi_year', type=int)
    epi_week = request.args.get('epi_week', type=int)

    query = DengueCase.query.filter_by(regency_id=regency_id)

    if epi_year:
        query = query.filter_by(epi_year=epi_year)
    if epi_week:
        query = query.filter_by(epi_week=epi_week)

    cases = query.order_by(DengueCase.epi_year.desc(), DengueCase.epi_week.desc()).all()

    return jsonify([{
        'epi_year': c.epi_year,
        'epi_week': c.epi_week,
        'cases': c.cases,
        'date': f"{c.epi_year}-W{c.epi_week:02d}"
    } for c in cases])


@public.route('/api/predictions/<int:regency_id>')
def api_predictions(regency_id):
    """API endpoint: Get predictions for a regency"""
    predictions = Prediction.query.filter_by(
        regency_id=regency_id
    ).order_by(Prediction.epi_year.desc(), Prediction.epi_week.desc()).all()

    return jsonify([{
        'epi_year': p.epi_year,
        'epi_week': p.epi_week,
        'predicted_cases': p.predicted_cases,
        'risk_level': p.risk_level,
        'confidence_lower': p.confidence_lower,
        'confidence_upper': p.confidence_upper,
        'date': f"{p.epi_year}-W{p.epi_week:02d}"
    } for p in predictions])
