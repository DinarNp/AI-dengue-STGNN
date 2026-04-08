"""
Admin Routes
Routes for admin users to manage data and system
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_file
from flask_login import current_user
import os
from datetime import datetime
from werkzeug.utils import secure_filename

from ..models import db, Regency, DengueCase, ClimateData, NDVIData, ModelVersion, DataProcessingLog
from ..services.auth import admin_required
from ..services.data_pipeline import DataPipelineService
from ..services.prediction import PredictionService
from ..services.training import TrainingService

admin = Blueprint('admin', __name__, url_prefix='/admin')


@admin.route('/dashboard')
@admin_required
def dashboard():
    """Admin dashboard"""
    # Get statistics
    stats = {
        'total_regencies': Regency.query.filter_by(is_active=True).count(),
        'total_cases': db.session.query(db.func.sum(DengueCase.cases)).scalar() or 0,
        'total_users': db.session.query(db.func.count(db.distinct('users.id'))).scalar() or 0,
        'latest_data_month': db.session.query(
            db.func.max(DengueCase.year * 12 + DengueCase.month)
        ).scalar()
    }
    
    # Get recent processing logs
    recent_logs = DataProcessingLog.query.order_by(
        DataProcessingLog.started_at.desc()
    ).limit(10).all()
    
    # Get active model
    active_model = ModelVersion.query.filter_by(is_active=True).first()
    
    return render_template('admin/dashboard.html',
                         stats=stats,
                         recent_logs=recent_logs,
                         active_model=active_model)


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
        ).order_by(DengueCase.year.desc(), DengueCase.month.desc()).first()

        latest_climate = ClimateData.query.filter_by(
            regency_id=regency.id
        ).order_by(ClimateData.year.desc(), ClimateData.month.desc()).first()

        latest_ndvi = NDVIData.query.filter_by(
            regency_id=regency.id
        ).order_by(NDVIData.year.desc(), NDVIData.month.desc()).first()

        # --- Overall completeness: months with ALL 3 types present ---
        # Collect distinct (year, month) sets per data type for this regency
        dengue_months = set(
            (r.year, r.month) for r in
            DengueCase.query.filter_by(regency_id=regency.id)
            .with_entities(DengueCase.year, DengueCase.month).all()
        )
        climate_months = set(
            (r.year, r.month) for r in
            ClimateData.query.filter_by(regency_id=regency.id)
            .with_entities(ClimateData.year, ClimateData.month).all()
        )
        ndvi_months = set(
            (r.year, r.month) for r in
            NDVIData.query.filter_by(regency_id=regency.id)
            .with_entities(NDVIData.year, NDVIData.month).all()
        )

        all_months = dengue_months | climate_months | ndvi_months
        complete_months = dengue_months & climate_months & ndvi_months

        completeness_pct = (
            round(len(complete_months) / len(all_months) * 100)
            if all_months else 0
        )

        data_stats.append({
            'regency_id': regency.id,
            'regency': regency.name,
            'latest_dengue': f"{latest_dengue.year}-{latest_dengue.month:02d}" if latest_dengue else 'N/A',
            'latest_climate': f"{latest_climate.year}-{latest_climate.month:02d}" if latest_climate else 'N/A',
            'latest_ndvi': f"{latest_ndvi.year}-{latest_ndvi.month:02d}" if latest_ndvi else 'N/A',
            'completeness_pct': completeness_pct,
        })
    
    return render_template('admin/data_management.html',
                         regencies=regencies,
                         data_stats=data_stats,
                         now=datetime.now())


@admin.route('/data/dengue/add', methods=['POST'])
@admin_required
def add_dengue_data():
    """Add or update dengue case data"""
    try:
        data = request.get_json()
        regency_id = int(data['regency_id'])
        year       = int(data['year'])
        month      = int(data['month'])
        cases      = int(data['cases'])
        notes      = data.get('notes', '')

        existing = DengueCase.query.filter_by(
            regency_id=regency_id, year=year, month=month
        ).first()

        if existing:
            existing.cases = cases
            existing.notes = notes
            existing.updated_at = datetime.utcnow()
            existing.reported_by_id = current_user.id
            action = 'updated'
        else:
            db.session.add(DengueCase(
                regency_id=regency_id,
                year=year,
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
    Fetch climate data for all regencies for a specific month
    ONE-CLICK OPERATION replacing the manual API calls
    """
    try:
        data = request.get_json()
        year = int(data['year'])
        month = int(data['month'])
        
        pipeline = DataPipelineService(current_app.config)
        result = pipeline.fetch_all_climate_data(year, month, current_user.id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/ndvi/upload', methods=['POST'])
@admin_required
def upload_ndvi_data():
    """
    Upload and process NDVI GeoTIFF file
    REPLACES manual NEO processing
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        year = int(request.form['year'])
        month = int(request.form['month'])
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Process NDVI data
        pipeline = DataPipelineService(current_app.config)
        result = pipeline.process_ndvi_from_satellite(upload_path, year, month, current_user.id)
        
        # Clean up uploaded file
        os.remove(upload_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/ndvi/impute', methods=['POST'])
@admin_required
def impute_ndvi():
    """Fill missing NDVI values using imputation"""
    try:
        data = request.get_json()
        year = int(data['year'])
        month = int(data['month'])
        
        pipeline = DataPipelineService(current_app.config)
        result = pipeline.impute_missing_ndvi(year, month)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/data/export', methods=['POST'])
@admin_required
def export_data():
    """
    Export data to CSV format
    CREATES the data_monthly_5kab_YYYY_YYYY_ndvi.csv file
    """
    try:
        data = request.get_json()
        year_start = int(data['year_start'])
        year_end = int(data['year_end'])
        
        # Create filename
        filename = f"data_monthly_5kab_{year_start}_{year_end}_ndvi.csv"
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
    """Generate predictions for all regencies"""
    try:
        data = request.get_json()
        year = int(data['year'])
        month = int(data['month'])
        
        prediction_service = PredictionService(current_app.config)
        result = prediction_service.predict_all_regencies(year, month)
        
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
            DengueCase.year.desc(), DengueCase.month.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
    elif data_type == 'climate':
        q = ClimateData.query
        if regency_id:
            q = q.filter_by(regency_id=regency_id)
        records = q.order_by(
            ClimateData.year.desc(), ClimateData.month.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
    elif data_type == 'ndvi':
        q = NDVIData.query
        if regency_id:
            q = q.filter_by(regency_id=regency_id)
        records = q.order_by(
            NDVIData.year.desc(), NDVIData.month.desc()
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
        fieldnames = ['Year', 'Month', 'Region', 'Cases', 'Notes']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name in regencies:
            writer.writerow({'Year': 2024, 'Month': 1, 'Region': name, 'Cases': 0, 'Notes': ''})

    elif data_type == 'climate':
        fieldnames = ['Year', 'Month', 'Region', 'Temperature_Min', 'Temperature_Max',
                      'Temperature_Avg', 'Humidity', 'Precipitation_Total',
                      'Pressure', 'Wind_Speed', 'Wind_Direction', 'Cloud_Cover']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name in regencies:
            writer.writerow({'Year': 2024, 'Month': 1, 'Region': name,
                             'Temperature_Min': '', 'Temperature_Max': '', 'Temperature_Avg': '',
                             'Humidity': '', 'Precipitation_Total': '', 'Pressure': '',
                             'Wind_Speed': '', 'Wind_Direction': '', 'Cloud_Cover': ''})

    elif data_type == 'ndvi':
        fieldnames = ['Year', 'Month', 'Region', 'NDVI', 'Is_Imputed']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name in regencies:
            writer.writerow({'Year': 2024, 'Month': 1, 'Region': name, 'NDVI': '', 'Is_Imputed': 'No'})

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
        Prediction.year,
        Prediction.month,
        db.func.count(Prediction.id).label('count'),
        db.func.avg(Prediction.predicted_cases).label('avg_predicted')
    ).group_by(
        Prediction.year,
        Prediction.month
    ).order_by(
        Prediction.year.desc(),
        Prediction.month.desc()
    ).limit(12).all()
    
    # Get active model
    active_model = ModelVersion.query.filter_by(is_active=True).first()
    
    return render_template('admin/predictions_manage.html',
                         recent_predictions=recent_predictions,
                         active_model=active_model)


@admin.route('/predictions/generate-batch', methods=['POST'])
@admin_required
def generate_batch_predictions():
    """Generate predictions for multiple months"""
    try:
        data = request.get_json()
        year = int(data['year'])
        months = data.get('months', [])  # List of month numbers
        
        prediction_service = PredictionService(current_app.config)
        
        results = []
        for month in months:
            result = prediction_service.predict_all_regencies(year, int(month))
            results.append({
                'year': year,
                'month': month,
                'result': result
            })
        
        total_success = sum(r['result']['success'] for r in results)
        total_predictions = sum(len(r['result'].get('predictions', [])) for r in results)
        
        return jsonify({
            'success': True,
            'total_predictions': total_predictions,
            'months_processed': len(results),
            'results': results,
            'message': f'Generated {total_predictions} predictions for {len(results)} months'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@admin.route('/predictions/export')
@admin_required
def export_predictions():
    """Export all predictions to CSV"""
    try:
        from ..models import Prediction
        
        predictions = Prediction.query.order_by(
            Prediction.year.desc(),
            Prediction.month.desc()
        ).all()
        
        records = []
        for pred in predictions:
            records.append({
                'Year': pred.year,
                'Month': pred.month,
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


# Import pandas for exports
import pandas as pd
