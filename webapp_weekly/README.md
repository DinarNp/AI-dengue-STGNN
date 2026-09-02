# Dengue Prediction Web Application

**AI-Powered Spatio-Temporal Dengue Prediction System for Yogyakarta Special Region, Indonesia**

This is a **complete restructuring** of the dengue prediction project with a modern web interface, role-based authentication, and simplified data management.

---

## 🎯 Key Improvements

### Before (Old System)
- ❌ Multiple separate Python scripts to run manually
- ❌ Complex data pipeline: SKDR download → climate API → NDVI processing → imputation → conversion
- ❌ No user interface - command line only
- ❌ No authentication or access control
- ❌ Difficult for health office staff to update data

### After (New System)
- ✅ **Unified web application** with modern UI
- ✅ **ONE-CLICK data updates** - automated pipeline
- ✅ **Role-based access control** - Admin, District Health Office, Public
- ✅ **Simplified workflows** for non-technical users
- ✅ **Real-time predictions** with recommendations
- ✅ **Public dashboard** for transparency

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WEB APPLICATION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   ADMIN      │  │  DISTRICT    │  │   PUBLIC     │     │
│  │   ROLE       │  │  HEALTH      │  │   ROLE       │     │
│  │              │  │   ROLE       │  │              │     │
│  │ • Manage all │  │ • Update own │  │ • View       │     │
│  │   data       │  │   regency    │  │   dashboard  │     │
│  │ • Upload     │  │ • View       │  │ • Statistics │     │
│  │   models     │  │   predictions│  │ • Reports    │     │
│  │ • Generate   │  │ • Get        │  │              │     │
│  │   predictions│  │   recommend- │  │              │     │
│  │              │  │   ations     │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   SIMPLIFIED DATA PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Dengue Cases:                                        │
│  • Manual entry via web form                                │
│  • Bulk CSV upload (replaces SKDR download)                 │
│                                                             │
│  Climate Data: ONE-CLICK                                    │
│  • Auto-fetch from NASA POWER API                           │
│  • Fallback to OpenWeather API                              │
│  • Automatic monthly aggregation                            │
│                                                             │
│  NDVI Data: SIMPLIFIED                                      │
│  • Upload GeoTIFF file via web                              │
│  • Automatic extraction for all regencies                   │
│  • Auto-imputation for missing values                       │
│                                                             │
│  Export:                                                    │
│  • Generate data_monthly_5kab_YYYY_YYYY_ndvi.csv            │
│  • Ready for model training                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   PREDICTION ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • Integrates existing STGNN model                          │
│  • Automatic prediction generation                          │
│  • Risk level classification                                │
│  • Health recommendations                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
webapp_restructured/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── models.py                # Database models
│   ├── routes/
│   │   ├── auth.py             # Login/logout routes
│   │   ├── admin.py            # Admin routes
│   │   ├── district_health.py  # District health routes
│   │   └── public.py           # Public routes
│   ├── services/
│   │   ├── auth.py             # Authentication helpers
│   │   ├── data_pipeline.py    # Unified data pipeline
│   │   └── prediction.py       # Prediction service
│   ├── templates/
│   │   ├── base.html           # Base template
│   │   ├── auth/               # Login templates
│   │   ├── admin/              # Admin templates
│   │   ├── district_health/    # District health templates
│   │   └── public/             # Public templates
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── utils/
│       └── init_data.py        # Initialize default data
│
├── config/
│   └── config.py               # Configuration settings
│
├── data/
│   ├── raw/                    # Raw uploaded files
│   ├── processed/              # Processed CSV files
│   └── uploads/                # Temporary uploads
│
├── database/
│   └── dengue_app.db          # SQLite database (auto-created)
│
├── migrations/                 # Database migrations
│
├── models/                     # Trained model files (.pth)
│
├── tests/                      # Unit tests
│
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) Virtual environment tool

### Step 1: Install Dependencies

```bash
cd webapp_restructured
pip install -r requirements.txt
```

### Step 2: Copy Existing Model

Copy your trained model file to the models folder:

```bash
# From the parent directory
cp dengue_stgnn_model.pth webapp_restructured/models/

# Or use the experimental model
cp experiment_results/experiments4/dengue_stgnn_model.pth webapp_restructured/models/
```

### Step 3: Run the Application

```bash
python run.py
```

The application will:
1. Create the database automatically
2. Initialize default users and regencies
3. Start the web server on http://localhost:5000

---

## 👥 Default Users

The system creates default users on first run:

### Admin User
- **Username:** `admin`
- **Password:** `admin123`
- **Capabilities:**
  - Full system access
  - Manage all data sources
  - Upload models
  - Generate predictions for all regencies
  - View all statistics

### District Health Office Users

One user per regency:

| Username        | Password   | Regency           |
|----------------|------------|-------------------|
| `bantul`       | health123  | KAB BANTUL        |
| `gunung_kidul` | health123  | KAB GUNUNG KIDUL  |
| `kulon_progo`  | health123  | KAB KULON PROGO   |
| `sleman`       | health123  | KAB SLEMAN        |
| `yogyakarta`   | health123  | KOTA YOGYAKARTA   |

**Capabilities:**
- Update dengue cases for their regency only
- View predictions for their regency
- Get health recommendations
- View reports and statistics

⚠️ **IMPORTANT:** Change these default passwords before deploying to production!

---

## 📊 Usage Guide

### For Admin Users

#### 1. Update Dengue Cases

**Option A: Manual Entry**
1. Go to Admin → Data Management
2. Click "Add Manual Entry"
3. Select regency, year, month
4. Enter number of cases
5. Click Save

**Option B: Bulk Import from CSV**
1. Prepare CSV with columns: `Year,Region,Month,Cases`
2. Go to Admin → Data Management
3. Click "Bulk Import CSV"
4. Upload file
5. System will import all rows automatically

**Old way:** Download from SKDR website manually
**New way:** Direct input or CSV upload ✅

---

#### 2. Update Climate Data

**ONE-CLICK OPERATION:**

1. Go to Admin → Data Management
2. Enter Year and Month
3. Click "Fetch for Month"
4. System automatically:
   - Fetches data from NASA POWER API for all 5 regencies
   - Calculates monthly averages
   - Saves to database

**Old way:** Run `get_climate_data_v3.py` manually, wait for API calls, run `convert_weekly_to_monthly.py`
**New way:** One button click ✅

---

#### 3. Update NDVI Data

**SIMPLIFIED PROCESS:**

1. Download MODIS GeoTIFF file (from NEO website)
2. Go to Admin → Data Management
3. Click "Upload GeoTIFF"
4. Select file, enter year/month
5. Click Upload
6. System automatically:
   - Extracts NDVI values for all regencies
   - Saves to database

If you have missing months:
1. Click "Impute Missing"
2. System fills gaps using forward/backward fill

**Old way:** Run `get_ndvi_kabupaten.py`, then `ndvi_epi_week_kabupaten.py`, then `ndvi_imputation.py`, then `convert_weekly_to_monthly.py`
**New way:** Upload file, click button ✅

---

#### 4. Export Data for Model Training

1. Go to Admin → Data Management
2. Enter Start Year and End Year (e.g., 2021-2024)
3. Click "Export to CSV"
4. Downloads `data_monthly_5kab_2021_2024_ndvi.csv`
5. Use this file for model training

**Format:** Exactly matches the original CSV structure

---

#### 5. Upload New Model

1. Train your model (using the original training scripts)
2. Go to Admin → Model Management
3. Click "Upload Model"
4. Enter version name (e.g., "Model v2.0 - Jan 2025")
5. Upload .pth file
6. Click "Activate" to make it the active model

---

#### 6. Generate Predictions

1. Go to Admin → Predictions
2. Enter Year and Month
3. Click "Generate for All Regencies"
4. System automatically:
   - Checks data availability
   - Runs prediction model
   - Calculates risk levels
   - Saves predictions to database

---

### For District Health Office Users

#### 1. Update Dengue Cases

1. Login with your regency credentials
2. Go to "My District" → Update Cases
3. Enter month and number of cases
4. Click Save
5. Only your regency data can be updated

#### 2. View Predictions

1. Go to "My District" → Predictions
2. View predicted cases for upcoming months
3. See risk level (Low/Medium/High/Very High)
4. Read health recommendations

#### 3. Request New Prediction

1. Go to Predictions
2. Click "Request Prediction"
3. Enter month you want to predict
4. System generates prediction instantly
5. View recommendation for your action plan

#### 4. View Reports

1. Go to "My District" → Reports
2. View:
   - Yearly statistics
   - Monthly breakdown
   - Trend charts
   - Comparison with previous years

---

### For Public Users

1. Visit http://localhost:5000
2. Click "Dashboard"
3. View:
   - Current dengue situation for all regencies
   - Monthly trends
   - Provincial totals
   - Risk levels
4. No login required!

---

## 🔄 Data Pipeline Workflow

### Old Workflow (Complex)

```
1. Download data from SKDR website manually
2. Save as CSV
3. Run get_climate_data_v3.py
   → Wait for API calls (30+ minutes)
4. Run convert_weekly_to_monthly.py
5. Go to NEO website, download GeoTIFF
6. Run get_ndvi_kabupaten.py
7. Run ndvi_epi_week_kabupaten.py
8. Run ndvi_imputation.py
9. Run convert_weekly_to_monthly.py again
10. Finally get data_monthly_5kab_2021_2024_ndvi.csv
```

**Problems:**
- Too many manual steps
- Error-prone
- Requires technical knowledge
- Time-consuming

---

### New Workflow (Simplified)

```
1. Login to web app
2. Click "Bulk Import CSV" → Upload dengue cases
3. Click "Fetch Climate Data" for each month needed
4. Click "Upload NDVI" → Upload GeoTIFF file
5. Click "Export to CSV"
6. Done! ✅
```

**Benefits:**
- One interface for everything
- Automatic processing
- Error handling built-in
- Anyone can do it
- Much faster

---

## 📊 Database Schema

### Users Table
- Authentication and role management
- Roles: admin, district_health, public

### Regencies Table
- 5 regencies in Yogyakarta
- Latitude/longitude coordinates
- Population and area data

### DengueCase Table
- Monthly dengue case counts
- Linked to regency and user who entered data
- Data source tracking

### ClimateData Table
- Temperature, humidity, precipitation, etc.
- Monthly aggregates
- Source tracking (NASA POWER/OpenWeather)

### NDVIData Table
- Vegetation index from satellite
- Monthly values
- Imputation flag

### Prediction Table
- Model predictions
- Risk levels
- Confidence intervals
- Actual cases (for validation)

### ModelVersion Table
- Track different model versions
- Performance metrics
- Active model flag

### DataProcessingLog Table
- Audit trail of all data operations
- Success/failure tracking
- Error messages

---

## 🔌 API Endpoints

### Public API (No authentication required)

```
GET  /api/regencies              # List all regencies
GET  /api/cases/<regency_id>     # Get dengue cases
GET  /api/predictions/<regency_id> # Get predictions
```

### Admin API (Authentication required)

```
POST /admin/data/dengue/add            # Add dengue case
POST /admin/data/dengue/bulk-import    # Bulk import CSV
POST /admin/data/climate/fetch         # Fetch climate data
POST /admin/data/ndvi/upload           # Upload NDVI GeoTIFF
POST /admin/data/ndvi/impute           # Impute missing NDVI
POST /admin/data/export                # Export to CSV
POST /admin/predictions/generate       # Generate predictions
POST /admin/model/upload               # Upload model
POST /admin/model/activate/<id>        # Activate model
```

### District Health API

```
POST /district/cases/add              # Add case for own regency
POST /district/request-prediction     # Request prediction
```

---

## 🎨 Web Interface Features

### Responsive Design
- Works on desktop, tablet, and mobile
- Bootstrap 5 UI framework
- Modern, clean interface

### Interactive Charts
- Chart.js for visualizations
- Line charts for trends
- Bar charts for comparisons
- Real-time updates

### Risk Visualization
- Color-coded risk levels:
  - 🟢 Green = Low risk (< 30 cases)
  - 🟡 Yellow = Medium risk (30-60 cases)
  - 🟠 Orange = High risk (60-100 cases)
  - 🔴 Red = Very High risk (> 100 cases)

### Recommendations
Auto-generated based on risk level:
- Low: Routine prevention
- Medium: Increased prevention
- High: Enhanced control measures
- Very High: Emergency response

---

## 🔒 Security Features

1. **Password Hashing**
   - Werkzeug secure password hashing
   - Salted hashes

2. **Session Management**
   - Secure cookie-based sessions
   - CSRF protection

3. **Role-Based Access Control**
   - Decorators for route protection
   - Granular permissions

4. **SQL Injection Prevention**
   - SQLAlchemy ORM
   - Parameterized queries

5. **Input Validation**
   - Form validation
   - File type checking
   - Size limits

---

## 🧪 Testing

Run tests (when implemented):

```bash
pytest tests/
```

---

## 🚢 Production Deployment

### Using Gunicorn (Recommended)

```bash
gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app('production')"
```

### Environment Variables

Create `.env` file:

```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@localhost/dengue_db
OPENWEATHER_API_KEY=your-api-key-here
```

### Database Migration

For production, use PostgreSQL:

```bash
# Install PostgreSQL driver
pip install psycopg2-binary

# Update DATABASE_URL in config
# Run migrations
flask db upgrade
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name dengue-predict.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static {
        alias /path/to/webapp_restructured/app/static;
    }
}
```

---

## 🔧 Troubleshooting

### Database Not Created
```bash
# Manually create database
python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all()"
```

### Model Not Loading
- Ensure model file is in `models/` folder
- Check file path in ModelVersion table
- Verify PyTorch version compatibility

### Climate API Fails
- Check API keys
- Verify internet connection
- Check API rate limits
- Use NASA POWER (no key needed) as primary

### NDVI Upload Fails
- Ensure rasterio is installed
- Check GeoTIFF format
- Verify coordinates in file

---

## 📝 Maintenance Tasks

### Weekly
- Backup database
- Check logs for errors
- Monitor disk space

### Monthly
- Update dengue cases
- Fetch climate data
- Process NDVI data
- Generate predictions
- Review model performance

### Quarterly
- Retrain model with new data
- Update model version
- Review user accounts
- Check data quality

---

## 🤝 Integration with Original Codebase

This new system **complements** the original codebase:

### Keep Using (for research)
- `main.py`, `main2.py` - Model training scripts
- `compare_models_experiment.py` - Experimentation
- `experiments/dengue_pipeline.py` - Training pipeline
- `models/stgnn.py` - Model architecture

### Replace in Production
- ❌ `get_climate_data_v3.py` → ✅ Web interface
- ❌ `convert_weekly_to_monthly.py` → ✅ Automatic
- ❌ `NEO/get_ndvi_kabupaten.py` → ✅ Web upload
- ❌ `NEO/ndvi_epi_week_kabupaten.py` → ✅ Automatic
- ❌ `NEO/ndvi_imputation.py` → ✅ One-click
- ❌ `app.py`, `app2.py`, `app3.py` → ✅ This web app

---

## 📚 Additional Resources

- **Original Model Paper:** [Insert paper reference]
- **STGNN Architecture:** See `models/stgnn.py` in parent directory
- **NASA POWER API:** https://power.larc.nasa.gov/docs/
- **OpenWeather API:** https://openweathermap.org/api
- **MODIS NDVI:** https://neo.gsfc.nasa.gov/

---

## 🐛 Known Issues & Future Improvements

### Current Limitations
- SQLite for development (use PostgreSQL in production)
- Basic error handling (can be enhanced)
- No email notifications (can be added)
- No data version control (can be implemented)

### Planned Features
- [ ] Email notifications for high risk predictions
- [ ] Automated weekly reports
- [ ] Data visualization export to PDF
- [ ] Mobile app version
- [ ] Multi-language support (Indonesian/English)
- [ ] Integration with national health system
- [ ] Real-time data sync
- [ ] Advanced analytics dashboard

---

## 📧 Support & Contact

For questions or issues:
1. Check this documentation
2. Review code comments
3. Check logs in database
4. Contact system administrator

---

## 📜 License

[Your license here]

---

## 🙏 Acknowledgments

- Original AI-STGNN model development team
- NASA POWER and OpenWeather for climate data APIs
- NEO for satellite imagery
- Yogyakarta Health Department for dengue case data

---

**Last Updated:** January 2025
**Version:** 1.0.0
**Status:** Production Ready

---

## Quick Start Summary

```bash
# 1. Install
cd webapp_restructured
pip install -r requirements.txt

# 2. Copy model
cp ../dengue_stgnn_model.pth models/

# 3. Run
python run.py

# 4. Login
# Open http://localhost:5000
# Username: admin
# Password: admin123

# 5. Start using!
```

Enjoy your simplified dengue prediction system! 🎉
