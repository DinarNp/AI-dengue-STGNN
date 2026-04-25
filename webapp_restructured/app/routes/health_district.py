"""
District Health Office Routes
Routes for district health office users
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import current_user
from datetime import datetime

from ..models import db, Regency, DengueCase, Prediction, ClimateData, NDVIData
from ..services.auth import health_district_required, check_regency_access
from ..i18n import get_lang, make_t
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
    
    current_year = datetime.now().year

    # Last 24 months of actual cases
    recent_cases = DengueCase.query.filter_by(
        regency_id=regency.id
    ).order_by(DengueCase.year.desc(), DengueCase.month.desc()).limit(24).all()

    # Latest prediction (for stat card)
    latest_prediction = Prediction.query.filter_by(
        regency_id=regency.id
    ).order_by(Prediction.year.desc(), Prediction.month.desc()).first()

    # Total cases this year
    total_cases_this_year = db.session.query(
        db.func.sum(DengueCase.cases)
    ).filter(
        DengueCase.regency_id == regency.id,
        DengueCase.year == current_year
    ).scalar() or 0

    # Build actual monthly trend (oldest → newest)
    monthly_trend = [
        {'month': f"{c.year}-{c.month:02d}", 'cases': c.cases}
        for c in reversed(recent_cases)
    ]

    # Next 3 predicted months after the latest actual month
    if recent_cases:
        latest_year  = recent_cases[0].year
        latest_month = recent_cases[0].month
    else:
        latest_year  = current_year
        latest_month = datetime.now().month

    prediction_trend = []
    py, pm = latest_year, latest_month
    for _ in range(3):
        pm += 1
        if pm > 12:
            pm = 1
            py += 1
        pred = Prediction.query.filter_by(
            regency_id=regency.id,
            year=py,
            month=pm
        ).first()
        prediction_trend.append({
            'month': f"{py}-{pm:02d}",
            'cases': round(float(pred.predicted_cases), 1) if pred else None
        })

    return render_template('health_district/dashboard.html',
                         regency=regency,
                         recent_cases=recent_cases,
                         latest_prediction=latest_prediction,
                         total_cases_this_year=total_cases_this_year,
                         monthly_trend=monthly_trend,
                         prediction_trend=prediction_trend,
                         current_year=current_year)


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


@health_district.route('/risk_monitor')
@health_district_required
def view_risk_monitor():
    """Risk Monitor: 3-month dengue forecast for user's regency"""
    from flask import current_app as _ca
    from sqlalchemy import func as _func

    regency = Regency.query.filter_by(name=current_user.regency).first()
    if not regency:
        flash('Your regency assignment is invalid. Please contact admin.', 'danger')
        return redirect(url_for('main.index'))

    prediction_service = PredictionService(_ca.config)

    _month_keys = ['january','february','march','april','may','june',
                   'july','august','september','october','november','december']
    _lang = get_lang()
    _t = make_t(_lang)

    # Latest actual data month for this regency
    latest_case = DengueCase.query.filter_by(regency_id=regency.id)\
        .order_by(DengueCase.year.desc(), DengueCase.month.desc()).first()
    if latest_case:
        latest_year, latest_month = latest_case.year, latest_case.month
    else:
        now = datetime.now()
        latest_year, latest_month = now.year, now.month

    # Build 3 forecast month entries
    forecast_months = []
    py, pm = latest_year, latest_month
    for _ in range(3):
        pm += 1
        if pm > 12:
            pm = 1
            py += 1

        pred = Prediction.query.filter_by(
            regency_id=regency.id, year=py, month=pm
        ).first()

        # Climate/NDVI: use same calendar month from previous year as proxy
        climate = ClimateData.query.filter_by(
            regency_id=regency.id, year=py - 1, month=pm
        ).first()
        ndvi = NDVIData.query.filter_by(
            regency_id=regency.id, year=py - 1, month=pm
        ).first()

        # Historical comparison: same month last year
        last_year_case = DengueCase.query.filter_by(
            regency_id=regency.id, year=py - 1, month=pm
        ).first()

        # Previous actual month (for month-over-month context)
        prev_pm = pm - 1 if pm > 1 else 12
        prev_py = py if pm > 1 else py - 1
        prev_case = DengueCase.query.filter_by(
            regency_id=regency.id, year=prev_py, month=prev_pm
        ).first()

        # Provincial average prediction for this month
        prov_total = db.session.query(_func.sum(Prediction.predicted_cases)).filter(
            Prediction.year == py, Prediction.month == pm
        ).scalar()
        prov_count = db.session.query(_func.count(Prediction.id)).filter(
            Prediction.year == py, Prediction.month == pm
        ).scalar()
        prov_avg = round(float(prov_total) / prov_count, 1) if (prov_total and prov_count) else None

        RECOMMENDATIONS = {
            'id': {
                'no_alert': [
                    "Lanjutkan pengendalian vektor rutin dan inspeksi rumah tangga",
                    "Promosi kesehatan reguler dan edukasi berbasis sekolah",
                    "Pemeliharaan sistem surveilans untuk deteksi dini",
                    "Kegiatan kesiapsiagaan musiman menjelang transisi musim hujan",
                    "Pelibatan masyarakat dalam pengelolaan lingkungan jangka panjang"
                ],
                'alert': [
                    "Prediksi kasus melebihi ambang batas endemik — segera aktifkan respons",
                    "Intensifikasi pengendalian vektor (pengelolaan sumber larva, fogging di area padat)",
                    "Pengerahan tim respons cepat untuk deteksi kasus aktif dan pelacakan kontak",
                    "Penguatan surveilans berbasis masyarakat melalui kader terlatih",
                    "Koordinasi fasilitas kesehatan untuk kesiapan penanganan kasus dan penimbunan sumber daya"
                ],
            },
            'en': {
                'no_alert': [
                    "Continuation of routine vector control and household inspections",
                    "Regular health promotion and school-based education",
                    "Surveillance system maintenance for early warning detection",
                    "Seasonal preparedness activities during pre-wet season transition",
                    "Community engagement in long-term environmental management"
                ],
                'alert': [
                    "Predicted cases exceed endemic threshold — activate response immediately",
                    "Intensify vector control (larval source management, fogging in high-density areas)",
                    "Deploy rapid response teams for active case detection and contact tracing",
                    "Strengthen community-based surveillance through trained volunteers",
                    "Coordinate healthcare facilities for case management readiness and resource stockpiling"
                ],
            },
        }

        if pred:
            recommendations = RECOMMENDATIONS.get(_lang, RECOMMENDATIONS.get('id', {})).get(pred.risk_level, [])

            # Year-over-year change
            if last_year_case and last_year_case.cases > 0:
                yoy = round(((pred.predicted_cases - last_year_case.cases) / last_year_case.cases) * 100, 1)
            else:
                yoy = None

            # Month-over-month vs last actual
            if prev_case and prev_case.cases > 0:
                mom = round(((pred.predicted_cases - prev_case.cases) / prev_case.cases) * 100, 1)
            else:
                mom = None

            explanation = _risk_explanation(pred.predicted_cases, last_year_case, climate, ndvi, _lang)
            regional    = _regional_analysis(regency.name, pred.predicted_cases, prov_avg, last_year_case, _lang)
        else:
            recommendations = []
            yoy = mom = explanation = regional = None

        forecast_months.append({
            'year': py,
            'month': pm,
            'month_name': _t('month.' + _month_keys[pm - 1]),
            'prediction': pred,
            'climate': climate,
            'ndvi': ndvi,
            'last_year_cases': last_year_case.cases if last_year_case else None,
            'prev_cases': prev_case.cases if prev_case else None,
            'prov_avg': prov_avg,
            'yoy_change': yoy,
            'mom_change': mom,
            'recommendations': recommendations,
            'explanation': explanation,
            'regional_analysis': regional,
        })

    risk_counts = {'alert': 0, 'no_alert': 0}
    for fm in forecast_months:
        if fm['prediction']:
            rl = fm['prediction'].risk_level
            risk_counts[rl] = risk_counts.get(rl, 0) + 1

    # Last 24 months actual trend (oldest → newest)
    recent_cases = DengueCase.query.filter_by(
        regency_id=regency.id
    ).order_by(DengueCase.year.desc(), DengueCase.month.desc()).limit(24).all()
    monthly_trend = [
        {'month': f"{c.year}-{c.month:02d}", 'cases': c.cases}
        for c in reversed(recent_cases)
    ]

    # 3 predicted months for the chart
    prediction_trend = [
        {
            'month': f"{fm['year']}-{fm['month']:02d}",
            'cases': round(float(fm['prediction'].predicted_cases), 1) if fm['prediction'] else None
        }
        for fm in forecast_months
    ]

    return render_template('health_district/risk_monitor.html',
                           regency=regency,
                           forecast_months=forecast_months,
                           risk_counts=risk_counts,
                           latest_year=latest_year,
                           latest_month=latest_month,
                           monthly_trend=monthly_trend,
                           prediction_trend=prediction_trend)


def _risk_explanation(predicted_cases, last_year_case, climate, ndvi, lang='id'):
    if lang == 'en':
        parts = [f"The AI model predicts {predicted_cases:.0f} dengue cases for this month."]
        if last_year_case and last_year_case.cases > 0:
            chg = ((predicted_cases - last_year_case.cases) / last_year_case.cases) * 100
            if chg > 20:
                parts.append(f"This is {chg:.0f}% higher than the same month last year "
                             f"({last_year_case.cases} cases), indicating an elevated risk trend.")
            elif chg < -20:
                parts.append(f"This is {abs(chg):.0f}% lower than the same month last year "
                             f"({last_year_case.cases} cases), suggesting improving conditions.")
            else:
                parts.append(f"This is comparable to the same month last year ({last_year_case.cases} cases).")
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
            parts.append("Environmental context is based on the same calendar month from the previous year.")
    else:
        parts = [f"Model AI memprediksi {predicted_cases:.0f} kasus demam berdarah untuk bulan ini."]
        if last_year_case and last_year_case.cases > 0:
            chg = ((predicted_cases - last_year_case.cases) / last_year_case.cases) * 100
            if chg > 20:
                parts.append(f"Angka ini {chg:.0f}% lebih tinggi dibanding bulan yang sama tahun lalu "
                             f"({last_year_case.cases} kasus), mengindikasikan tren risiko yang meningkat.")
            elif chg < -20:
                parts.append(f"Angka ini {abs(chg):.0f}% lebih rendah dibanding bulan yang sama tahun lalu "
                             f"({last_year_case.cases} kasus), menunjukkan kondisi yang membaik.")
            else:
                parts.append(f"Angka ini sebanding dengan bulan yang sama tahun lalu ({last_year_case.cases} kasus).")
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
            parts.append("Konteks lingkungan didasarkan pada bulan kalender yang sama dari tahun sebelumnya.")
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
            parts.append(f"For reference, the same month last year recorded {last_year_case.cases} actual cases.")
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
            parts.append(f"Sebagai referensi, bulan yang sama tahun lalu mencatat {last_year_case.cases} kasus aktual.")
    return " ".join(parts)


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
    
    current_year = datetime.now().year

    # All years that have data
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

    selected_year = request.args.get('year', current_year, type=int)

    # Yearly statistics (all years with data)
    yearly_stats = []
    for year in sorted(set(available_years)):
        total_cases = db.session.query(
            db.func.sum(DengueCase.cases)
        ).filter(
            DengueCase.regency_id == regency.id,
            DengueCase.year == year
        ).scalar() or 0
        yearly_stats.append({'year': year, 'total_cases': total_cases})

    # Monthly breakdown for selected year
    monthly_breakdown = DengueCase.query.filter_by(
        regency_id=regency.id,
        year=selected_year
    ).order_by(DengueCase.month).all()

    return render_template('health_district/reports.html',
                         regency=regency,
                         yearly_stats=yearly_stats,
                         monthly_breakdown=monthly_breakdown,
                         available_years=available_years,
                         selected_year=selected_year,
                         current_year=current_year)
