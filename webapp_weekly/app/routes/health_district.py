"""
District Health Office Routes
Routes for district health office users
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import current_user
from datetime import datetime, date
from epiweeks import Week

from ..models import db, Regency, DengueCase, Prediction, ClimateData, NDVIData
from ..services.auth import health_district_required, check_regency_access
from ..i18n import get_lang, make_t
from ..services.data_pipeline import DataPipelineService
from ..services.prediction import PredictionService

health_district = Blueprint('health_district', __name__, url_prefix='/district')

# Trend charts and forecast cards are scoped to a fixed year rather than a
# rolling window: actual/reported data covers the year up to the last
# FORECAST_WEEKS_COUNT weeks, and the forecast covers those final weeks.
TARGET_YEAR = 2025
FORECAST_WEEKS_COUNT = 4
_WEEKS_IN_TARGET_YEAR = Week.fromdate(date(TARGET_YEAR, 12, 28), system='iso').week
FORECAST_START_WEEK = _WEEKS_IN_TARGET_YEAR - FORECAST_WEEKS_COUNT + 1
ACTUAL_END_WEEK = FORECAST_START_WEEK - 1


@health_district.route('/dashboard')
@health_district_required
def dashboard():
    """Health district office dashboard (weekly cadence), scoped to TARGET_YEAR:
    actual data covers weeks 1..ACTUAL_END_WEEK, forecast covers the final
    FORECAST_WEEKS_COUNT weeks."""
    regency = Regency.query.filter_by(name=current_user.regency).first()

    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))

    # Actual cases for TARGET_YEAR, weeks 1..ACTUAL_END_WEEK
    recent_cases = DengueCase.query.filter(
        DengueCase.regency_id == regency.id,
        DengueCase.epi_year == TARGET_YEAR,
        DengueCase.epi_week <= ACTUAL_END_WEEK
    ).order_by(DengueCase.epi_week).all()

    # Latest prediction (for stat card)
    latest_prediction = Prediction.query.filter_by(
        regency_id=regency.id
    ).order_by(Prediction.epi_year.desc(), Prediction.epi_week.desc()).first()

    # Total cases this epi-year
    total_cases_this_year = db.session.query(
        db.func.sum(DengueCase.cases)
    ).filter(
        DengueCase.regency_id == regency.id,
        DengueCase.epi_year == TARGET_YEAR
    ).scalar() or 0

    # Build actual weekly trend (oldest -> newest)
    weekly_trend = [
        {'week': f"{TARGET_YEAR}-W{c.epi_week:02d}", 'cases': c.cases}
        for c in recent_cases
    ]

    # Forecast: last FORECAST_WEEKS_COUNT weeks of TARGET_YEAR
    forecast_weeks = []
    prediction_trend = []
    for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
        pred = Prediction.query.filter_by(
            regency_id=regency.id, epi_year=TARGET_YEAR, epi_week=w
        ).first()
        forecast_weeks.append({'epi_year': TARGET_YEAR, 'epi_week': w, 'prediction': pred})
        prediction_trend.append({
            'week': f"{TARGET_YEAR}-W{w:02d}",
            'cases': round(float(pred.predicted_cases), 1) if pred else None
        })

    return render_template('health_district/dashboard.html',
                         regency=regency,
                         recent_cases=recent_cases,
                         latest_prediction=latest_prediction,
                         total_cases_this_year=total_cases_this_year,
                         weekly_trend=weekly_trend,
                         prediction_trend=prediction_trend,
                         forecast_weeks=forecast_weeks,
                         current_year=TARGET_YEAR)


@health_district.route('/update-cases')
@health_district_required
def update_cases():
    """View, add, and edit dengue cases for user's regency, by epidemiological week"""
    from epiweeks import Week

    regency = Regency.query.filter_by(name=current_user.regency).first()

    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))

    current_week = Week.thisweek(system='iso')
    current_year = current_week.year
    selected_year = request.args.get('year', current_year, type=int)

    # Cases for selected epi-year
    cases = DengueCase.query.filter_by(
        regency_id=regency.id,
        epi_year=selected_year
    ).order_by(DengueCase.epi_week).all()

    # All epi-years that have data (plus current year always available)
    year_rows = (
        db.session.query(DengueCase.epi_year)
        .filter_by(regency_id=regency.id)
        .distinct()
        .order_by(DengueCase.epi_year.desc())
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

    last_week_of_year = Week.fromdate(datetime(selected_year, 12, 28).date(), system='iso').week

    return render_template('health_district/update_cases.html',
                           regency=regency,
                           cases=cases,
                           selected_year=selected_year,
                           available_years=available_years,
                           total_selected_year=total_selected_year,
                           total_all_time=total_all_time,
                           current_epi_year=current_week.year,
                           current_epi_week=current_week.week,
                           last_week_of_year=last_week_of_year)


@health_district.route('/cases/add', methods=['POST'])
@health_district_required
def add_cases():
    """Add or update dengue cases for one epidemiological week — direct DB write"""
    try:
        data = request.get_json()
        regency = Regency.query.filter_by(name=current_user.regency).first()

        if not regency:
            return jsonify({'success': False, 'message': 'Invalid regency assignment'}), 403

        epi_year = int(data['epi_year'])
        epi_week = int(data['epi_week'])
        cases_count = int(data['cases'])
        notes = data.get('notes', '')

        from epiweeks import Week
        try:
            month = Week(epi_year, epi_week, system='iso').startdate().month
        except ValueError:
            return jsonify({'success': False, 'message': f'Invalid epi-week {epi_week} for year {epi_year}'}), 400

        existing = DengueCase.query.filter_by(
            regency_id=regency.id, epi_year=epi_year, epi_week=epi_week
        ).first()

        if existing:
            existing.cases = cases_count
            existing.notes = notes
            existing.month = month
            existing.updated_at = datetime.utcnow()
            existing.reported_by_id = current_user.id
            action = 'updated'
        else:
            db.session.add(DengueCase(
                regency_id=regency.id,
                epi_year=epi_year,
                epi_week=epi_week,
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


@health_district.route('/risk_monitor')
@health_district_required
def view_risk_monitor():
    """Risk Monitor: dengue forecast for user's regency over the final
    FORECAST_WEEKS_COUNT weeks of TARGET_YEAR, using the binary statistical
    endemic-channel classification (mean + 1.25*SD of this region's cases in
    the same epi-week across prior years), matching
    webapp_restructured/app/services/prediction.py::_calculate_risk_level."""
    from flask import current_app as _ca
    from sqlalchemy import func as _func
    import numpy as _np

    regency = Regency.query.filter_by(name=current_user.regency).first()
    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))

    prediction_service = PredictionService(_ca.config)
    _lang = get_lang()

    # Build forecast week entries for the final FORECAST_WEEKS_COUNT weeks of TARGET_YEAR
    forecast_weeks = []
    for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
        wk = Week(TARGET_YEAR, w, system='iso')

        pred = Prediction.query.filter_by(
            regency_id=regency.id, epi_year=wk.year, epi_week=wk.week
        ).first()

        climate = ClimateData.query.filter_by(regency_id=regency.id, epi_year=wk.year, epi_week=wk.week).first()
        ndvi = NDVIData.query.filter_by(regency_id=regency.id, epi_year=wk.year, epi_week=wk.week).first()

        # Historical comparison: same epi-week last year
        last_year_week = Week(wk.year - 1, wk.week, system='iso')
        last_year_case = DengueCase.query.filter_by(
            regency_id=regency.id, epi_year=last_year_week.year, epi_week=last_year_week.week
        ).first()

        # Previous actual week (for week-over-week context)
        prev_week = wk - 1
        prev_case = DengueCase.query.filter_by(
            regency_id=regency.id, epi_year=prev_week.year, epi_week=prev_week.week
        ).first()

        # Provincial average prediction for this week
        prov_total = db.session.query(_func.sum(Prediction.predicted_cases)).filter(
            Prediction.epi_year == wk.year, Prediction.epi_week == wk.week
        ).scalar()
        prov_count = db.session.query(_func.count(Prediction.id)).filter(
            Prediction.epi_year == wk.year, Prediction.epi_week == wk.week
        ).scalar()
        prov_avg = round(float(prov_total) / prov_count, 1) if (prov_total and prov_count) else None

        if pred:
            recommendation_text = prediction_service.get_recommendation(pred.predicted_cases, pred.risk_level)

            if last_year_case and last_year_case.cases > 0:
                yoy = round(((pred.predicted_cases - last_year_case.cases) / last_year_case.cases) * 100, 1)
            else:
                yoy = None

            if prev_case and prev_case.cases > 0:
                wow = round(((pred.predicted_cases - prev_case.cases) / prev_case.cases) * 100, 1)
            else:
                wow = None

            explanation = _risk_explanation(pred.predicted_cases, last_year_case, climate, ndvi, _lang)
            regional    = _regional_analysis(regency.name, pred.predicted_cases, prov_avg, last_year_case, _lang)
        else:
            recommendation_text = None
            yoy = wow = explanation = regional = None

        forecast_weeks.append({
            'epi_year': wk.year,
            'epi_week': wk.week,
            'prediction': pred,
            'climate': climate,
            'ndvi': ndvi,
            'last_year_cases': last_year_case.cases if last_year_case else None,
            'prev_cases': prev_case.cases if prev_case else None,
            'prov_avg': prov_avg,
            'yoy_change': yoy,
            'wow_change': wow,
            'recommendation': recommendation_text,
            'explanation': explanation,
            'regional_analysis': regional,
        })

    risk_counts = {'alert': 0, 'no_alert': 0}
    for fw in forecast_weeks:
        if fw['prediction']:
            rl = fw['prediction'].risk_level
            risk_counts[rl] = risk_counts.get(rl, 0) + 1

    # Actual trend for TARGET_YEAR, weeks 1..ACTUAL_END_WEEK (oldest -> newest)
    recent_cases = DengueCase.query.filter(
        DengueCase.regency_id == regency.id,
        DengueCase.epi_year == TARGET_YEAR,
        DengueCase.epi_week <= ACTUAL_END_WEEK
    ).order_by(DengueCase.epi_week).all()
    weekly_trend = [
        {'week': f"{TARGET_YEAR}-W{c.epi_week:02d}", 'cases': c.cases}
        for c in recent_cases
    ]

    prediction_trend = [
        {
            'week': f"{fw['epi_year']}-W{fw['epi_week']:02d}",
            'cases': round(float(fw['prediction'].predicted_cases), 1) if fw['prediction'] else None
        }
        for fw in forecast_weeks
    ]

    # Threshold line (mean + 1.25*SD) aligned with weekly_trend + prediction_trend,
    # matching the same-epi-week historical-lookup pattern used on the public dashboard.
    threshold_trend = []
    for item in weekly_trend:
        yoi, woi = item['week'].split('-W')
        yoi, woi = int(yoi), int(woi)
        hist_vals = db.session.query(DengueCase.cases).filter(
            DengueCase.regency_id == regency.id,
            DengueCase.epi_week == woi,
            DengueCase.epi_year >= 2021,
            DengueCase.epi_year <= yoi - 1
        ).all()
        vals = [r.cases for r in hist_vals if r.cases is not None]
        if len(vals) >= 2:
            hm = float(_np.mean(vals)); hs = float(_np.std(vals, ddof=1))
            threshold_trend.append(round(hm + 1.25 * hs, 1))
        elif vals:
            threshold_trend.append(round(float(vals[0]), 1))
        else:
            threshold_trend.append(None)
    for fw in forecast_weeks:
        pred = fw['prediction']
        threshold_trend.append(round(float(pred.alert_threshold), 1) if pred and pred.alert_threshold is not None else None)

    return render_template('health_district/risk_monitor.html',
                           regency=regency,
                           forecast_weeks=forecast_weeks,
                           risk_counts=risk_counts,
                           latest_epi_year=TARGET_YEAR,
                           latest_epi_week=ACTUAL_END_WEEK,
                           weekly_trend=weekly_trend,
                           prediction_trend=prediction_trend,
                           threshold_trend=threshold_trend)


def _risk_explanation(predicted_cases, last_year_case, climate, ndvi, lang='id'):
    if lang == 'en':
        parts = [f"The AI model predicts {predicted_cases:.0f} dengue cases for this week."]
        if last_year_case and last_year_case.cases > 0:
            chg = ((predicted_cases - last_year_case.cases) / last_year_case.cases) * 100
            if chg > 20:
                parts.append(f"This is {chg:.0f}% higher than the same epi-week last year "
                             f"({last_year_case.cases} cases), indicating an elevated risk trend.")
            elif chg < -20:
                parts.append(f"This is {abs(chg):.0f}% lower than the same epi-week last year "
                             f"({last_year_case.cases} cases), suggesting improving conditions.")
            else:
                parts.append(f"This is comparable to the same epi-week last year ({last_year_case.cases} cases).")
        env = []
        if climate:
            if climate.temperature_avg and 25 <= climate.temperature_avg <= 30:
                env.append(f"temperature ({climate.temperature_avg:.1f}°C) in the optimal range for mosquito breeding")
            if climate.precipitation_total and climate.precipitation_total > 100:
                env.append(f"high rainfall ({climate.precipitation_total:.0f} mm) creating breeding sites")
            elif climate.precipitation_total and climate.precipitation_total > 50:
                env.append(f"moderate rainfall ({climate.precipitation_total:.0f} mm)")
            if climate.humidity and climate.humidity > 80:
                env.append(f"high humidity ({climate.humidity:.0f}%) favouring mosquito survival")
        if ndvi and ndvi.ndvi_value and ndvi.ndvi_value > 0.4:
            env.append(f"dense vegetation (NDVI {ndvi.ndvi_value:.3f}) providing larval habitat")
        if env:
            parts.append("Contributing environmental factors: " + "; ".join(env) + ".")
        elif not climate:
            parts.append("Environmental context for this future week is not yet available.")
    else:
        parts = [f"Model AI memprediksi {predicted_cases:.0f} kasus demam berdarah untuk minggu ini."]
        if last_year_case and last_year_case.cases > 0:
            chg = ((predicted_cases - last_year_case.cases) / last_year_case.cases) * 100
            if chg > 20:
                parts.append(f"Angka ini {chg:.0f}% lebih tinggi dibanding minggu epidemiologi yang sama tahun lalu "
                             f"({last_year_case.cases} kasus), mengindikasikan tren risiko yang meningkat.")
            elif chg < -20:
                parts.append(f"Angka ini {abs(chg):.0f}% lebih rendah dibanding minggu epidemiologi yang sama tahun lalu "
                             f"({last_year_case.cases} kasus), menunjukkan kondisi yang membaik.")
            else:
                parts.append(f"Angka ini sebanding dengan minggu epidemiologi yang sama tahun lalu ({last_year_case.cases} kasus).")
        env = []
        if climate:
            if climate.temperature_avg and 25 <= climate.temperature_avg <= 30:
                env.append(f"suhu ({climate.temperature_avg:.1f}°C) dalam rentang optimal untuk perkembangbiakan nyamuk")
            if climate.precipitation_total and climate.precipitation_total > 100:
                env.append(f"curah hujan tinggi ({climate.precipitation_total:.0f} mm) menciptakan tempat perkembangbiakan")
            elif climate.precipitation_total and climate.precipitation_total > 50:
                env.append(f"curah hujan sedang ({climate.precipitation_total:.0f} mm)")
            if climate.humidity and climate.humidity > 80:
                env.append(f"kelembaban tinggi ({climate.humidity:.0f}%) mendukung kelangsungan hidup nyamuk")
        if ndvi and ndvi.ndvi_value and ndvi.ndvi_value > 0.4:
            env.append(f"vegetasi lebat (NDVI {ndvi.ndvi_value:.3f}) menyediakan habitat larva")
        if env:
            parts.append("Faktor lingkungan yang berkontribusi: " + "; ".join(env) + ".")
        elif not climate:
            parts.append("Konteks lingkungan untuk minggu mendatang ini belum tersedia.")
    return " ".join(parts)


def _regional_analysis(regency_name, predicted_cases, prov_avg, last_year_case, lang='id'):
    if lang == 'en':
        parts = [f"{regency_name} is forecast at {predicted_cases:.0f} cases."]
        if prov_avg:
            ratio = predicted_cases / prov_avg
            if ratio > 1.2:
                parts.append(f"This is {((ratio-1)*100):.0f}% above the DIY provincial average "
                             f"({prov_avg:.0f} cases), indicating a higher-than-average burden.")
            elif ratio < 0.8:
                parts.append(f"This is {((1-ratio)*100):.0f}% below the DIY provincial average "
                             f"({prov_avg:.0f} cases), suggesting relatively lower risk.")
            else:
                parts.append(f"This aligns with the DIY provincial average of {prov_avg:.0f} cases.")
        if last_year_case:
            parts.append(f"For reference, the same epi-week last year recorded {last_year_case.cases} actual cases.")
    else:
        parts = [f"{regency_name} diprakirakan mencapai {predicted_cases:.0f} kasus."]
        if prov_avg:
            ratio = predicted_cases / prov_avg
            if ratio > 1.2:
                parts.append(f"Angka ini {((ratio-1)*100):.0f}% di atas rata-rata provinsi DIY "
                             f"({prov_avg:.0f} kasus), mengindikasikan beban yang lebih tinggi dari rata-rata.")
            elif ratio < 0.8:
                parts.append(f"Angka ini {((1-ratio)*100):.0f}% di bawah rata-rata provinsi DIY "
                             f"({prov_avg:.0f} kasus), menunjukkan risiko yang relatif lebih rendah.")
            else:
                parts.append(f"Angka ini sejalan dengan rata-rata provinsi DIY sebesar {prov_avg:.0f} kasus.")
        if last_year_case:
            parts.append(f"Sebagai referensi, minggu epidemiologi yang sama tahun lalu mencatat {last_year_case.cases} kasus aktual.")
    return " ".join(parts)


@health_district.route('/request-prediction', methods=['POST'])
@health_district_required
def request_prediction():
    """Request new prediction for an upcoming epidemiological week"""
    try:
        data = request.get_json()

        # Get user's regency
        regency = Regency.query.filter_by(name=current_user.regency).first()

        if not regency:
            return jsonify({'success': False, 'message': 'Invalid regency assignment'}), 403

        epi_year = int(data['epi_year'])
        epi_week = int(data['epi_week'])

        # Generate prediction
        from flask import current_app as _ca; prediction_service = PredictionService(_ca.config)
        result = prediction_service.predict_single_regency(regency.id, epi_year, epi_week)
        
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
    """View reports and statistics (weekly cadence)"""
    from epiweeks import Week

    regency = Regency.query.filter_by(name=current_user.regency).first()

    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))

    current_year = Week.thisweek(system='iso').year

    # All epi-years that have data
    year_rows = (
        db.session.query(DengueCase.epi_year)
        .filter_by(regency_id=regency.id)
        .distinct()
        .order_by(DengueCase.epi_year.desc())
        .all()
    )
    available_years = [y[0] for y in year_rows]
    if current_year not in available_years:
        available_years.insert(0, current_year)

    selected_year = request.args.get('year', current_year, type=int)

    # Yearly statistics (all years with data)
    yearly_stats = []
    for year in sorted(set(available_years)):
        total_cases = db.session.query(
            db.func.sum(DengueCase.cases)
        ).filter(
            DengueCase.regency_id == regency.id,
            DengueCase.epi_year == year
        ).scalar() or 0
        yearly_stats.append({'year': year, 'total_cases': total_cases})

    # Weekly breakdown for selected epi-year
    weekly_breakdown = DengueCase.query.filter_by(
        regency_id=regency.id,
        epi_year=selected_year
    ).order_by(DengueCase.epi_week).all()

    return render_template('health_district/reports.html',
                         regency=regency,
                         yearly_stats=yearly_stats,
                         weekly_breakdown=weekly_breakdown,
                         available_years=available_years,
                         selected_year=selected_year,
                         current_year=current_year)
