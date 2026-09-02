"""
Admin Routes
Routes for admin users to manage data and system
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_file
from flask_login import current_user
import os
from datetime import datetime, date
from werkzeug.utils import secure_filename
from epiweeks import Week

from ..models import db, Regency, DengueCase, ClimateData, NDVIData, ModelVersion, DataProcessingLog, Prediction
from ..services.auth import admin_required
from ..i18n import get_lang, make_t
from ..services.data_pipeline import DataPipelineService
from ..services.prediction import PredictionService
from ..services.training import TrainingService

admin = Blueprint('admin', __name__, url_prefix='/admin')

# Trend charts and forecast cards are scoped to a fixed year rather than a
# rolling window: actual/reported data covers the year up to the last
# FORECAST_WEEKS_COUNT weeks, and the forecast covers those final weeks.
TARGET_YEAR = 2025
FORECAST_WEEKS_COUNT = 4
_WEEKS_IN_TARGET_YEAR = Week.fromdate(date(TARGET_YEAR, 12, 28), system='iso').week
FORECAST_START_WEEK = _WEEKS_IN_TARGET_YEAR - FORECAST_WEEKS_COUNT + 1
ACTUAL_END_WEEK = FORECAST_START_WEEK - 1


@admin.route('/dashboard')
@admin_required
def dashboard():
    """Admin dashboard. Trend charts scoped to TARGET_YEAR: actual data covers
    weeks 1..ACTUAL_END_WEEK, forecast covers the final FORECAST_WEEKS_COUNT weeks."""
    import numpy as _np

    # Get statistics
    latest_case_row = DengueCase.query.order_by(
        DengueCase.epi_year.desc(), DengueCase.epi_week.desc()
    ).first()
    stats = {
        'total_regencies': Regency.query.filter_by(is_active=True).count(),
        'total_cases': db.session.query(db.func.sum(DengueCase.cases)).scalar() or 0,
        'total_users': db.session.query(db.func.count(db.distinct('users.id'))).scalar() or 0,
        'latest_data_week': f"{latest_case_row.epi_year}-W{latest_case_row.epi_week:02d}" if latest_case_row else 'N/A',
    }

    # Get recent processing logs
    recent_logs = DataProcessingLog.query.order_by(
        DataProcessingLog.started_at.desc()
    ).limit(10).all()

    # Get active model
    active_model = ModelVersion.query.filter_by(is_active=True).first()

    # Build chart data per regency (TARGET_YEAR actual + forecast + threshold)
    regencies_list = Regency.query.filter_by(is_active=True).order_by(Regency.id).all()
    all_chart_data = []
    for reg in regencies_list:
        actual_cases = DengueCase.query.filter(
            DengueCase.regency_id == reg.id,
            DengueCase.epi_year == TARGET_YEAR,
            DengueCase.epi_week <= ACTUAL_END_WEEK
        ).order_by(DengueCase.epi_week).all()
        weekly_trend = [{'week': f"{TARGET_YEAR}-W{c.epi_week:02d}", 'cases': c.cases} for c in actual_cases]

        prediction_trend = []
        for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
            pred = Prediction.query.filter_by(regency_id=reg.id, epi_year=TARGET_YEAR, epi_week=w).first()
            prediction_trend.append({
                'week': f"{TARGET_YEAR}-W{w:02d}",
                'cases': round(float(pred.predicted_cases), 1) if pred else None
            })

        threshold_trend = []
        for item in weekly_trend:
            woi = int(item['week'].split('-W')[1])
            hist_vals = db.session.query(DengueCase.cases).filter(
                DengueCase.regency_id == reg.id,
                DengueCase.epi_week == woi,
                DengueCase.epi_year >= 2021,
                DengueCase.epi_year <= TARGET_YEAR - 1
            ).all()
            vals = [r.cases for r in hist_vals if r.cases is not None]
            if len(vals) >= 2:
                hm = float(_np.mean(vals))
                hs = float(_np.std(vals, ddof=1))
                threshold_trend.append(round(hm + 1.25 * hs, 1))
            elif vals:
                threshold_trend.append(round(float(vals[0]), 1))
            else:
                threshold_trend.append(None)
        for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
            pred2 = Prediction.query.filter_by(regency_id=reg.id, epi_year=TARGET_YEAR, epi_week=w).first()
            if pred2 and pred2.alert_threshold is not None:
                threshold_trend.append(round(float(pred2.alert_threshold), 1))
            else:
                threshold_trend.append(None)

        all_chart_data.append({
            'regency_id': reg.id,
            'regency_name': reg.name,
            'weekly_trend': weekly_trend,
            'prediction_trend': prediction_trend,
            'threshold_trend': threshold_trend,
        })

    return render_template('admin/dashboard.html',
                         stats=stats,
                         recent_logs=recent_logs,
                         active_model=active_model,
                         all_chart_data=all_chart_data)


@admin.route('/data-management')
@admin_required
def data_management():
    """Data management interface"""
    regencies = Regency.query.filter_by(is_active=True).all()
    
    # Get data completeness statistics
    data_stats = []
    for regency in regencies:
        latest_dengue = DengueCase.query.filter_by(
            regency_id=regency.id
        ).order_by(DengueCase.epi_year.desc(), DengueCase.epi_week.desc()).first()

        latest_climate = ClimateData.query.filter_by(
            regency_id=regency.id
        ).order_by(ClimateData.epi_year.desc(), ClimateData.epi_week.desc()).first()

        latest_ndvi = NDVIData.query.filter_by(
            regency_id=regency.id
        ).order_by(NDVIData.epi_year.desc(), NDVIData.epi_week.desc()).first()

        # --- Overall completeness: epi-weeks with ALL 3 types present ---
        # Collect distinct (epi_year, epi_week) sets per data type for this regency
        dengue_weeks = set(
            (r.epi_year, r.epi_week) for r in
            DengueCase.query.filter_by(regency_id=regency.id)
            .with_entities(DengueCase.epi_year, DengueCase.epi_week).all()
        )
        climate_weeks = set(
            (r.epi_year, r.epi_week) for r in
            ClimateData.query.filter_by(regency_id=regency.id)
            .with_entities(ClimateData.epi_year, ClimateData.epi_week).all()
        )
        ndvi_weeks = set(
            (r.epi_year, r.epi_week) for r in
            NDVIData.query.filter_by(regency_id=regency.id)
            .with_entities(NDVIData.epi_year, NDVIData.epi_week).all()
        )

        all_weeks = dengue_weeks | climate_weeks | ndvi_weeks
        complete_weeks = dengue_weeks & climate_weeks & ndvi_weeks

        completeness_pct = (
            round(len(complete_weeks) / len(all_weeks) * 100)
            if all_weeks else 0
        )

        data_stats.append({
            'regency_id': regency.id,
            'regency': regency.name,
            'latest_dengue': f"{latest_dengue.epi_year}-W{latest_dengue.epi_week:02d}" if latest_dengue else 'N/A',
            'latest_climate': f"{latest_climate.epi_year}-W{latest_climate.epi_week:02d}" if latest_climate else 'N/A',
            'latest_ndvi': f"{latest_ndvi.epi_year}-W{latest_ndvi.epi_week:02d}" if latest_ndvi else 'N/A',
            'completeness_pct': completeness_pct,
        })
    
    return render_template('admin/data_management.html',
                         regencies=regencies,
                         data_stats=data_stats,
                         now=datetime.now())


@admin.route('/data/dengue/add', methods=['POST'])
@admin_required
def add_dengue_data():
    """Add or update dengue case data for one epidemiological week"""
    try:
        data = request.get_json()
        regency_id = int(data['regency_id'])
        epi_year   = int(data['epi_year'])
        epi_week   = int(data['epi_week'])
        cases      = int(data['cases'])
        notes      = data.get('notes', '')

        from epiweeks import Week
        try:
            month = Week(epi_year, epi_week, system='iso').startdate().month
        except ValueError:
            return jsonify({'success': False, 'message': f'Invalid epi-week {epi_week} for year {epi_year}'}), 400

        existing = DengueCase.query.filter_by(
            regency_id=regency_id, epi_year=epi_year, epi_week=epi_week
        ).first()

        if existing:
            existing.cases = cases
            existing.notes = notes
            existing.month = month
            existing.updated_at = datetime.utcnow()
            existing.reported_by_id = current_user.id
            action = 'updated'
        else:
            db.session.add(DengueCase(
                regency_id=regency_id,
                epi_year=epi_year,
                epi_week=epi_week,
                month=month,
                cases=cases,
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


@admin.route('/data/dengue/<int:case_id>/edit', methods=['PATCH'])
@admin_required
def edit_dengue_case(case_id):
    """Edit dengue case value"""
    try:
        data = request.get_json()
        case = DengueCase.query.get_or_404(case_id)
        case.cases = int(data['cases'])
        case.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True, 'message': 'Case updated successfully', 'cases': case.cases})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/dengue/bulk-import', methods=['POST'])
@admin_required
def bulk_import_dengue():
    """Bulk import dengue cases from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Import data
        pipeline = DataPipelineService(current_app.config)
        result = pipeline.bulk_import_dengue_cases(upload_path, current_user.id)
        
        # Clean up uploaded file
        os.remove(upload_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/climate/fetch', methods=['POST'])
@admin_required
def fetch_climate_data():
    """
    Fetch climate data for all regencies for a specific epidemiological week
    ONE-CLICK OPERATION replacing the manual API calls
    """
    try:
        data = request.get_json()
        epi_year = int(data['epi_year'])
        epi_week = int(data['epi_week'])

        pipeline = DataPipelineService(current_app.config)
        result = pipeline.fetch_climate_data_for_week_all_regencies(epi_year, epi_week, current_user.id)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/ndvi/fetch', methods=['POST'])
@admin_required
def fetch_ndvi_data():
    """
    Fetch NDVI for all regencies for a specific epidemiological week from the
    NASA NEO CSV archive (automated -- no manual GeoTIFF upload needed).
    """
    try:
        from ..services.ndvi_weekly import fetch_ndvi_for_week

        data = request.get_json()
        epi_year = int(data['epi_year'])
        epi_week = int(data['epi_week'])

        result = fetch_ndvi_for_week(epi_year, epi_week)

        log = DataProcessingLog(
            user_id=current_user.id, process_type='ndvi_process',
            status='completed' if result['success'] else 'failed',
            records_processed=result['written'],
            details={'epi_year': epi_year, 'epi_week': epi_week, 'composite_date': result['composite_date']},
            completed_at=datetime.utcnow()
        )
        db.session.add(log)
        db.session.commit()

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/export', methods=['POST'])
@admin_required
def export_data():
    """
    Export data to CSV format
    CREATES the data_weekly_5kab_YYYY_YYYY_ndvi.csv file
    """
    try:
        data = request.get_json()
        year_start = int(data['year_start'])
        year_end = int(data['year_end'])

        # Create filename
        filename = f"data_weekly_5kab_{year_start}_{year_end}_ndvi.csv"
        output_path = os.path.join(current_app.config['PROCESSED_DATA_FOLDER'], filename)
        
        # Export data
        pipeline = DataPipelineService(current_app.config)
        result = pipeline.export_to_csv(year_start, year_end, output_path)
        
        if result['success']:
            return send_file(
                output_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/predictions/generate', methods=['POST'])
@admin_required
def generate_predictions():
    """Generate predictions for all regencies for one epidemiological week"""
    try:
        data = request.get_json()
        epi_year = int(data['epi_year'])
        epi_week = int(data['epi_week'])

        prediction_service = PredictionService(current_app.config)
        result = prediction_service.predict_all_regencies(epi_year, epi_week)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/model/upload', methods=['POST'])
@admin_required
def upload_model():
    """Upload new model file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        version_name = request.form['version_name']
        description = request.form.get('description', '')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        # Save model file
        filename = secure_filename(file.filename)
        model_path = os.path.join(current_app.config['MODEL_FOLDER'], filename)
        file.save(model_path)
        
        # Create model version record
        new_model = ModelVersion(
            version_name=version_name,
            model_file=model_path,
            description=description,
            training_date=datetime.utcnow()
        )
        db.session.add(new_model)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Model {version_name} uploaded successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/model/activate/<int:model_id>', methods=['POST'])
@admin_required
def activate_model(model_id):
    """Activate a model version"""
    try:
        # Deactivate all models
        ModelVersion.query.update({'is_active': False})
        
        # Activate selected model
        model = ModelVersion.query.get(model_id)
        if not model:
            return jsonify({'success': False, 'message': 'Model not found'}), 404
        
        model.is_active = True
        db.session.commit()
        
        # Reload model in prediction service
        prediction_service = PredictionService(current_app.config)
        prediction_service.load_active_model()
        
        return jsonify({
            'success': True,
            'message': f'Model {model.version_name} activated'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/logs')
@admin_required
def view_logs():
    """View data processing logs"""
    logs = DataProcessingLog.query.order_by(
        DataProcessingLog.started_at.desc()
    ).limit(100).all()
    
    return render_template('admin/logs.html', logs=logs)


# Import current_app
from flask import current_app


@admin.route('/training')
@admin_required
def training():
    """Model training interface"""
    training_service = TrainingService(current_app.config)
    status = training_service.get_training_status()
    
    # Get all models for selection
    all_models = ModelVersion.query.order_by(
        ModelVersion.training_date.desc()
    ).all()
    
    return render_template('admin/training.html',
                         status=status,
                         all_models=all_models)


@admin.route('/training/export-data', methods=['POST'])
@admin_required
def export_training_data():
    """Export data for model training"""
    try:
        data = request.get_json()
        year_start = int(data['year_start'])
        year_end = int(data['year_end'])
        
        # Create filename
        filename = f"training_data_{year_start}_{year_end}.csv"
        output_path = os.path.join(current_app.config['PROCESSED_DATA_FOLDER'], filename)
        
        # Export data
        training_service = TrainingService(current_app.config)
        result = training_service.export_training_data(year_start, year_end, output_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'filename': filename,
                'records': result['records'],
                'message': result['message']
            })
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/training/train-model', methods=['POST'])
@admin_required
def train_model():
    """Train a new model"""
    try:
        data = request.get_json()
        
        # Get training data file
        data_filename = data.get('data_filename')
        if not data_filename:
            return jsonify({
                'success': False,
                'message': 'Data filename is required'
            }), 400
        
        data_path = os.path.join(current_app.config['PROCESSED_DATA_FOLDER'], data_filename)
        
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'message': f'Data file not found: {data_filename}'
            }), 400
        
        model_name = data.get('model_name', f'Model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        description = data.get('description', '')
        
        # Start training
        training_service = TrainingService(current_app.config)
        result = training_service.train_model(
            data_csv_path=data_path,
            model_name=model_name,
            user_id=current_user.id,
            description=description
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 400


@admin.route('/training/activate-model/<int:model_id>', methods=['POST'])
@admin_required
def activate_training_model(model_id):
    """Activate a trained model"""
    try:
        training_service = TrainingService(current_app.config)
        result = training_service.activate_model(model_id)
        
        if result['success']:
            # Reload prediction service with new model
            prediction_service = PredictionService(current_app.config)
            prediction_service.load_active_model()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400


@admin.route('/training/delete-model/<int:model_id>', methods=['DELETE'])
@admin_required
def delete_model(model_id):
    """Delete a model version"""
    try:
        model = ModelVersion.query.get(model_id)
        
        if not model:
            return jsonify({
                'success': False,
                'message': 'Model not found'
            }), 404
        
        if model.is_active:
            return jsonify({
                'success': False,
                'message': 'Cannot delete active model. Please activate another model first.'
            }), 400
        
        # Delete model file
        if os.path.exists(model.model_file):
            os.remove(model.model_file)
        
        # Delete database record
        db.session.delete(model)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Model "{model.version_name}" deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400


# ==========================================
# DATA MANAGEMENT ROUTES
# ==========================================

@admin.route('/data/view/<data_type>')
@admin_required
def view_data(data_type):
    """View data table with pagination"""
    page = request.args.get('page', 1, type=int)
    regency_id = request.args.get('regency_id', type=int)
    per_page = 50

    regency = Regency.query.get(regency_id) if regency_id else None

    if data_type == 'dengue':
        q = DengueCase.query
        if regency_id:
            q = q.filter_by(regency_id=regency_id)
        records = q.order_by(
            DengueCase.epi_year.desc(), DengueCase.epi_week.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
    elif data_type == 'climate':
        q = ClimateData.query
        if regency_id:
            q = q.filter_by(regency_id=regency_id)
        records = q.order_by(
            ClimateData.epi_year.desc(), ClimateData.epi_week.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
    elif data_type == 'ndvi':
        q = NDVIData.query
        if regency_id:
            q = q.filter_by(regency_id=regency_id)
        records = q.order_by(
            NDVIData.epi_year.desc(), NDVIData.epi_week.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
    else:
        flash('Invalid data type', 'danger')
        return redirect(url_for('admin.data_management'))

    return render_template('admin/data_view.html',
                         data_type=data_type,
                         records=records,
                         regency=regency,
                         regency_id=regency_id)


@admin.route('/data/template/<data_type>')
@admin_required
def download_template(data_type):
    """Return a CSV template for the given data type with correct headers and regency names"""
    import io
    import csv
    from flask import Response

    regencies = [r.name for r in Regency.query.filter_by(is_active=True).order_by(Regency.id).all()]

    output = io.StringIO()

    if data_type == 'dengue':
        fieldnames = ['Epi_Year', 'Epi_Week', 'Region', 'Cases', 'Notes']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name in regencies:
            writer.writerow({'Epi_Year': 2024, 'Epi_Week': 1, 'Region': name, 'Cases': 0, 'Notes': ''})

    elif data_type == 'climate':
        fieldnames = ['Epi_Year', 'Epi_Week', 'Region', 'Temperature_Min', 'Temperature_Max',
                      'Temperature_Avg', 'Humidity', 'Precipitation_Total',
                      'Pressure', 'Wind_Speed', 'Wind_Direction', 'Cloud_Cover']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name in regencies:
            writer.writerow({'Epi_Year': 2024, 'Epi_Week': 1, 'Region': name,
                             'Temperature_Min': '', 'Temperature_Max': '', 'Temperature_Avg': '',
                             'Humidity': '', 'Precipitation_Total': '', 'Pressure': '',
                             'Wind_Speed': '', 'Wind_Direction': '', 'Cloud_Cover': ''})

    elif data_type == 'ndvi':
        fieldnames = ['Epi_Year', 'Epi_Week', 'Region', 'NDVI', 'Is_Imputed']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name in regencies:
            writer.writerow({'Epi_Year': 2024, 'Epi_Week': 1, 'Region': name, 'NDVI': '', 'Is_Imputed': 'No'})

    else:
        return jsonify({'success': False, 'message': 'Invalid data type'}), 400

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={data_type}_template.csv'}
    )


@admin.route('/data/export-csv/<data_type>')
@admin_required
def export_csv(data_type):
    """Export data to CSV file"""
    from ..services.csv_manager import CSVDataManager
    
    csv_manager = CSVDataManager(current_app.config)
    
    if data_type == 'dengue':
        result = csv_manager.export_dengue_cases()
    elif data_type == 'climate':
        result = csv_manager.export_climate_data()
    elif data_type == 'ndvi':
        result = csv_manager.export_ndvi_data()
    else:
        return jsonify({'success': False, 'message': 'Invalid data type'}), 400
    
    if result['success']:
        return send_file(result['filepath'], as_attachment=True)
    else:
        return jsonify(result), 400


@admin.route('/data/import-csv/<data_type>', methods=['POST'])
@admin_required
def import_csv(data_type):
    """Import data from CSV file"""
    from ..services.csv_manager import CSVDataManager
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        # Save uploaded file
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Import data
        csv_manager = CSVDataManager(current_app.config)
        
        if data_type == 'dengue':
            result = csv_manager.import_dengue_cases(upload_path, current_user.id)
        elif data_type == 'climate':
            result = csv_manager.import_climate_data(upload_path)
        elif data_type == 'ndvi':
            result = csv_manager.import_ndvi_data(upload_path)
        else:
            return jsonify({'success': False, 'message': 'Invalid data type'}), 400
        
        # Clean up uploaded file
        os.remove(upload_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/validate', methods=['POST'])
@admin_required
def validate_data():
    """Validate data completeness for training"""
    from ..services.csv_manager import CSVDataManager
    
    try:
        data = request.get_json()
        year_start = int(data.get('year_start', 2021))
        year_end = int(data.get('year_end', 2024))
        
        csv_manager = CSVDataManager(current_app.config)
        result = csv_manager.validate_data_completeness(year_start, year_end)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/merge-for-training', methods=['POST'])
@admin_required
def merge_for_training():
    """Merge data for training"""
    from ..services.csv_manager import CSVDataManager
    
    try:
        data = request.get_json()
        year_start = int(data.get('year_start', 2021))
        year_end = int(data.get('year_end', 2024))
        
        csv_manager = CSVDataManager(current_app.config)
        result = csv_manager.merge_data_for_training(year_start, year_end)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


# ==========================================
# PREDICTION MANAGEMENT ROUTES
# ==========================================

@admin.route('/predictions-manage')
@admin_required
def predictions_manage():
    """Prediction management interface"""
    from ..models import Prediction
    
    # Get recent predictions
    recent_predictions = db.session.query(
        Prediction.epi_year,
        Prediction.epi_week,
        db.func.count(Prediction.id).label('count'),
        db.func.avg(Prediction.predicted_cases).label('avg_predicted')
    ).group_by(
        Prediction.epi_year,
        Prediction.epi_week
    ).order_by(
        Prediction.epi_year.desc(),
        Prediction.epi_week.desc()
    ).limit(12).all()
    
    # Get active model
    active_model = ModelVersion.query.filter_by(is_active=True).first()
    
    return render_template('admin/predictions_manage.html',
                         recent_predictions=recent_predictions,
                         active_model=active_model)


@admin.route('/predictions/generate-next-months', methods=['POST'])
@admin_required
def generate_next_months():
    """Generate predictions for the next N epi-weeks after a given epi-week"""
    try:
        data = request.get_json()
        from_epi_year = int(data['from_epi_year'])
        from_epi_week = int(data['from_epi_week'])
        n_weeks       = int(data.get('n_weeks', 3))

        prediction_service = PredictionService(current_app.config)
        result = prediction_service.predict_next_n_weeks(from_epi_year, from_epi_week, n_weeks)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/predictions/generate-batch', methods=['POST'])
@admin_required
def generate_batch_predictions():
    """Generate predictions for an arbitrary list of epi-weeks in one epi-year"""
    try:
        data = request.get_json()
        epi_year = int(data['epi_year'])
        epi_weeks = data.get('epi_weeks', [])  # List of epi-week numbers (1-53)

        prediction_service = PredictionService(current_app.config)

        results = []
        for epi_week in epi_weeks:
            result = prediction_service.predict_all_regencies(epi_year, int(epi_week))
            results.append({
                'epi_year': epi_year,
                'epi_week': epi_week,
                'result': result
            })

        total_predictions = sum(len(r['result'].get('predictions', [])) for r in results)

        return jsonify({
            'success': True,
            'total_predictions': total_predictions,
            'weeks_processed': len(results),
            'results': results,
            'message': f'Generated {total_predictions} predictions for {len(results)} weeks'
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/predictions/delete', methods=['DELETE'])
@admin_required
def delete_predictions():
    """Delete all predictions for a given epi-year/epi-week"""
    try:
        from ..models import Prediction
        data = request.get_json()
        epi_year = int(data['epi_year'])
        epi_week = int(data['epi_week'])

        deleted = Prediction.query.filter_by(epi_year=epi_year, epi_week=epi_week).delete()
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Deleted {deleted} prediction(s) for {epi_year}-W{epi_week:02d}'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/predictions/export')
@admin_required
def export_predictions():
    """Export all predictions to CSV"""
    try:
        from ..models import Prediction

        predictions = Prediction.query.order_by(
            Prediction.epi_year.desc(),
            Prediction.epi_week.desc()
        ).all()

        records = []
        for pred in predictions:
            records.append({
                'Epi_Year': pred.epi_year,
                'Epi_Week': pred.epi_week,
                'Region': pred.regency.name,
                'Predicted_Cases': pred.predicted_cases,
                'Zero_Probability': pred.zero_probability,
                'Confidence_Lower': pred.confidence_lower,
                'Confidence_Upper': pred.confidence_upper,
                'Risk_Level': pred.risk_level,
                'Actual_Cases': pred.actual_cases,
                'Model_Version': pred.model_version,
                'Prediction_Date': pred.prediction_date.strftime('%Y-%m-%d') if pred.prediction_date else ''
            })
        
        df = pd.DataFrame(records)
        export_path = os.path.join(
            current_app.config['PROCESSED_DATA_FOLDER'],
            f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        df.to_csv(export_path, index=False)
        
        return send_file(export_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


RISK_RECOMMENDATIONS = {
    'id': {
        'no_alert': [
            "Lanjutkan inspeksi rutin dan eliminasi tempat perkembangbiakan nyamuk (3M Plus)",
            "Pertahankan edukasi kesehatan masyarakat rutin tentang pencegahan demam berdarah",
            "Pastikan pemantauan mingguan terhadap potensi tempat perkembangbiakan",
            "Jaga surveilans vektor tetap aktif melalui program Jumantik",
        ],
        'alert': [
            "Prediksi kasus melebihi ambang batas endemik — segera aktifkan respons wabah",
            "Segera laksanakan fogging terarah di area terdampak dan sekitarnya",
            "Kerahkan tim respons cepat untuk penemuan kasus aktif dan pelacakan kontak",
            "Pastikan ketersediaan darah dan trombosit yang memadai di rumah sakit setempat",
            "Keluarkan peringatan kesehatan masyarakat melalui semua saluran media yang tersedia",
            "Koordinasikan dengan pemerintah daerah dan dinas kesehatan provinsi",
            "Eskalasi ke status KLB jika jumlah kasus terus meningkat",
        ],
    },
    'en': {
        'no_alert': [
            "Continue regular inspection and elimination of mosquito breeding sites (3M Plus)",
            "Maintain routine community health education on dengue prevention",
            "Ensure weekly monitoring of potential breeding sites",
            "Keep vector surveillance active through Jumantik program",
        ],
        'alert': [
            "Predicted cases exceed endemic threshold — activate outbreak response immediately",
            "Implement targeted fogging in affected and surrounding areas",
            "Deploy rapid response teams for active case finding and contact tracing",
            "Ensure adequate blood and platelet supplies at local hospitals",
            "Issue public health alerts through all available media channels",
            "Coordinate with local government and provincial health office",
            "Escalate to outbreak status if case count continues to rise",
        ],
    },
}

_MONTH_KEYS = ['january','february','march','april','may','june',
               'july','august','september','october','november','december']


def _build_regency_forecast(regency):
    """Build forecast data for a single regency, covering the final
    FORECAST_WEEKS_COUNT weeks of TARGET_YEAR."""
    from sqlalchemy import func as _func

    forecast_weeks = []
    for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
        wk = Week(TARGET_YEAR, w, system='iso')
        py, pw = wk.year, wk.week

        pred = Prediction.query.filter_by(regency_id=regency.id, epi_year=py, epi_week=pw).first()
        climate = ClimateData.query.filter_by(regency_id=regency.id, epi_year=py, epi_week=pw).first()
        ndvi = NDVIData.query.filter_by(regency_id=regency.id, epi_year=py, epi_week=pw).first()

        last_year_week = Week(py - 1, pw, system='iso')
        last_year_case = DengueCase.query.filter_by(
            regency_id=regency.id, epi_year=last_year_week.year, epi_week=last_year_week.week
        ).first()

        prev_week = wk - 1
        prev_case = DengueCase.query.filter_by(
            regency_id=regency.id, epi_year=prev_week.year, epi_week=prev_week.week
        ).first()

        prov_total = db.session.query(_func.sum(Prediction.predicted_cases)).filter(
            Prediction.epi_year == py, Prediction.epi_week == pw).scalar()
        prov_count = db.session.query(_func.count(Prediction.id)).filter(
            Prediction.epi_year == py, Prediction.epi_week == pw).scalar()
        prov_avg = round(float(prov_total) / prov_count, 1) if (prov_total and prov_count) else None

        lang = get_lang()
        if pred:
            recommendations = RISK_RECOMMENDATIONS.get(lang, RISK_RECOMMENDATIONS['en']).get(pred.risk_level, [])
            yoy = round(((pred.predicted_cases - last_year_case.cases) / last_year_case.cases) * 100, 1) \
                if (last_year_case and last_year_case.cases > 0) else None
            wow = round(((pred.predicted_cases - prev_case.cases) / prev_case.cases) * 100, 1) \
                if (prev_case and prev_case.cases > 0) else None
            explanation = _risk_explanation_admin(pred.predicted_cases, last_year_case, climate, ndvi, lang)
            regional    = _regional_analysis_admin(regency.name, pred.predicted_cases, prov_avg, last_year_case, lang)
        else:
            recommendations = []
            yoy = wow = explanation = regional = None

        forecast_weeks.append({
            'epi_year': py, 'epi_week': pw,
            'prediction': pred, 'climate': climate, 'ndvi': ndvi,
            'last_year_cases': last_year_case.cases if last_year_case else None,
            'prev_cases': prev_case.cases if prev_case else None,
            'prov_avg': prov_avg,
            'yoy_change': yoy, 'wow_change': wow,
            'recommendations': recommendations,
            'explanation': explanation,
            'regional_analysis': regional,
        })

    risk_counts = {'alert': 0, 'no_alert': 0}
    for fm in forecast_weeks:
        if fm['prediction']:
            rl = fm['prediction'].risk_level
            risk_counts[rl] = risk_counts.get(rl, 0) + 1

    risk_level = 'alert' if risk_counts.get('alert', 0) > 0 else \
                 ('no_alert' if risk_counts.get('no_alert', 0) > 0 else None)

    import numpy as _np

    actual_cases = DengueCase.query.filter(
        DengueCase.regency_id == regency.id,
        DengueCase.epi_year == TARGET_YEAR,
        DengueCase.epi_week <= ACTUAL_END_WEEK
    ).order_by(DengueCase.epi_week).all()
    weekly_trend = [{'week': f"{TARGET_YEAR}-W{c.epi_week:02d}", 'cases': c.cases} for c in actual_cases]
    prediction_trend = [
        {'week': f"{fm['epi_year']}-W{fm['epi_week']:02d}",
         'cases': round(float(fm['prediction'].predicted_cases), 1) if fm['prediction'] else None}
        for fm in forecast_weeks
    ]

    # Threshold (mean + 1.25 SD) for every week on the chart (historical + forecast)
    # Uses only same-epi-week data from 2021 up to TARGET_YEAR-1.
    threshold_trend = []
    for item in weekly_trend:
        week_of_item = int(item['week'].split('-W')[1])
        hist_vals = db.session.query(DengueCase.cases).filter(
            DengueCase.regency_id == regency.id,
            DengueCase.epi_week == week_of_item,
            DengueCase.epi_year >= 2021,
            DengueCase.epi_year <= TARGET_YEAR - 1
        ).all()
        vals = [r.cases for r in hist_vals if r.cases is not None]
        if len(vals) >= 2:
            hm = float(_np.mean(vals))
            hs = float(_np.std(vals, ddof=1))
            threshold_trend.append(round(hm + 1.25 * hs, 1))
        elif vals:
            threshold_trend.append(round(float(vals[0]), 1))
        else:
            threshold_trend.append(None)
    for fm in forecast_weeks:
        pred = fm['prediction']
        if pred and pred.alert_threshold is not None:
            threshold_trend.append(round(float(pred.alert_threshold), 1))
        else:
            threshold_trend.append(None)

    return {
        'regency': regency,
        'latest_epi_year': TARGET_YEAR,
        'latest_epi_week': ACTUAL_END_WEEK,
        'forecast_weeks': forecast_weeks,
        'risk_counts': risk_counts,
        'risk_level': risk_level,
        'weekly_trend': weekly_trend,
        'prediction_trend': prediction_trend,
        'threshold_trend': threshold_trend,
    }


@admin.route('/risk-monitor')
@admin_required
def risk_monitor():
    """Provincial risk monitor overview — compact card per regency, covering
    the final FORECAST_WEEKS_COUNT weeks of TARGET_YEAR."""
    regencies = Regency.query.filter_by(is_active=True).order_by(Regency.id).all()
    regency_forecasts = []

    for regency in regencies:
        # Risk summary + trend for overview, over the fixed forecast window
        forecast_weeks_summary = []
        for w in range(FORECAST_START_WEEK, _WEEKS_IN_TARGET_YEAR + 1):
            pred = Prediction.query.filter_by(regency_id=regency.id, epi_year=TARGET_YEAR, epi_week=w).first()
            forecast_weeks_summary.append({
                'epi_year': TARGET_YEAR, 'epi_week': w,
                'prediction': pred,
            })

        risk_counts = {'alert': 0, 'no_alert': 0}
        for fm in forecast_weeks_summary:
            if fm['prediction']:
                rl = fm['prediction'].risk_level
                risk_counts[rl] = risk_counts.get(rl, 0) + 1

        risk_level = 'alert' if risk_counts.get('alert', 0) > 0 else \
                     ('no_alert' if risk_counts.get('no_alert', 0) > 0 else None)

        import numpy as _np_rm
        recent_cases = DengueCase.query.filter(
            DengueCase.regency_id == regency.id,
            DengueCase.epi_year == TARGET_YEAR,
            DengueCase.epi_week <= ACTUAL_END_WEEK
        ).order_by(DengueCase.epi_week).all()
        weekly_trend = [{'week': f"{TARGET_YEAR}-W{c.epi_week:02d}", 'cases': c.cases} for c in recent_cases]
        prediction_trend = [
            {'week': f"{fm['epi_year']}-W{fm['epi_week']:02d}",
             'cases': round(float(fm['prediction'].predicted_cases), 1) if fm['prediction'] else None}
            for fm in forecast_weeks_summary
        ]

        threshold_trend = []
        for item in weekly_trend:
            woi = int(item['week'].split('-W')[1])
            hist_vals = db.session.query(DengueCase.cases).filter(
                DengueCase.regency_id == regency.id,
                DengueCase.epi_week == woi,
                DengueCase.epi_year >= 2021,
                DengueCase.epi_year <= TARGET_YEAR - 1
            ).all()
            vals = [r.cases for r in hist_vals if r.cases is not None]
            if len(vals) >= 2:
                hm = float(_np_rm.mean(vals))
                hs = float(_np_rm.std(vals, ddof=1))
                threshold_trend.append(round(hm + 1.25 * hs, 1))
            elif vals:
                threshold_trend.append(round(float(vals[0]), 1))
            else:
                threshold_trend.append(None)
        for fm in forecast_weeks_summary:
            pred_t = fm['prediction']
            if pred_t and pred_t.alert_threshold is not None:
                threshold_trend.append(round(float(pred_t.alert_threshold), 1))
            else:
                threshold_trend.append(None)

        regency_forecasts.append({
            'regency': regency,
            'latest_epi_year': TARGET_YEAR,
            'latest_epi_week': ACTUAL_END_WEEK,
            'forecast_weeks': forecast_weeks_summary,
            'risk_counts': risk_counts,
            'risk_level': risk_level,
            'weekly_trend': weekly_trend,
            'prediction_trend': prediction_trend,
            'threshold_trend': threshold_trend,
        })

    province_risk = {'alert': 0, 'no_alert': 0}
    for rf in regency_forecasts:
        if rf['risk_level']:
            province_risk[rf['risk_level']] = province_risk.get(rf['risk_level'], 0) + 1

    return render_template('admin/risk_monitor.html',
                           regency_forecasts=regency_forecasts,
                           province_risk=province_risk)


@admin.route('/risk-monitor/<int:regency_id>')
@admin_required
def risk_monitor_detail(regency_id):
    """Full 3-week risk analysis for a single regency"""
    regency = Regency.query.get_or_404(regency_id)
    rf = _build_regency_forecast(regency)
    _t_rmd = make_t(get_lang())
    return render_template('admin/risk_monitor_detail.html',
                           rf=rf,
                           month_names=[_t_rmd('month.' + k) for k in _MONTH_KEYS])


def _risk_explanation_admin(predicted_cases, last_year_case, climate, ndvi, lang='id'):
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


def _regional_analysis_admin(regency_name, predicted_cases, prov_avg, last_year_case, lang='id'):
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


# Import pandas for exports
import pandas as pd
