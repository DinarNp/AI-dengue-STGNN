# System Architecture

## 🏗️ Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WEB BROWSER (Frontend)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Bootstrap 5 UI │ Chart.js │ Font Awesome │ jQuery │ Responsive Design │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓↑ HTTP/HTTPS
┌─────────────────────────────────────────────────────────────────────────┐
│                        FLASK WEB APPLICATION                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                          ROUTES LAYER                             │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │  auth.py        │  admin.py      │  district_health.py  │ public.py│ │
│  │  - Login        │  - Data Mgmt   │  - Update Cases      │ - Dashboard│
│  │  - Logout       │  - Model Mgmt  │  - View Predictions  │ - Stats  │
│  │                 │  - Predictions │  - Reports           │ - API    │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    ↓↑                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        SERVICES LAYER                             │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │  DataPipelineService    │  PredictionService  │  AuthService     │ │
│  │  - Add dengue cases     │  - Load model       │  - Login check   │ │
│  │  - Fetch climate (API)  │  - Prepare data     │  - Role check    │ │
│  │  - Process NDVI         │  - Generate pred    │  - Permissions   │ │
│  │  - Impute missing       │  - Risk level       │                  │ │
│  │  - Export CSV           │  - Recommendations  │                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    ↓↑                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        DATABASE LAYER (ORM)                       │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │                         SQLAlchemy Models                         │ │
│  │  User │ Regency │ DengueCase │ ClimateData │ NDVIData │ Prediction│ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓↑
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATABASE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  SQLite (Development) │ PostgreSQL (Production)                        │
└─────────────────────────────────────────────────────────────────────────┘

                                    ↓↑
┌─────────────────────────────────────────────────────────────────────────┐
│                       EXTERNAL SERVICES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  NASA POWER API │ OpenWeather API │ MODIS NEO │ STGNN Model (.pth)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Application Structure

```
webapp_restructured/
│
├── 🏃 run.py                          # Entry point - Start here!
│
├── 📁 app/                            # Main application package
│   │
│   ├── __init__.py                    # App factory, blueprint registration
│   ├── models.py                      # 8 database models (370 lines)
│   │
│   ├── 📁 routes/                     # URL routing (4 blueprints)
│   │   ├── auth.py                   # /auth/* - Login, logout
│   │   ├── admin.py                  # /admin/* - Admin features
│   │   ├── district_health.py        # /district/* - District health
│   │   └── public.py                 # /* - Public pages, API
│   │
│   ├── 📁 services/                   # Business logic
│   │   ├── auth.py                   # Authentication decorators
│   │   ├── data_pipeline.py          # Data processing (700 lines)
│   │   └── prediction.py             # ML prediction (300 lines)
│   │
│   ├── 📁 templates/                  # HTML templates (Jinja2)
│   │   ├── base.html                 # Base layout
│   │   ├── 📁 auth/
│   │   │   └── login.html
│   │   ├── 📁 admin/
│   │   │   ├── dashboard.html
│   │   │   ├── data_management.html
│   │   │   └── logs.html
│   │   ├── 📁 district_health/
│   │   │   ├── dashboard.html
│   │   │   ├── update_cases.html
│   │   │   └── predictions.html
│   │   └── 📁 public/
│   │       ├── index.html
│   │       ├── dashboard.html
│   │       └── statistics.html
│   │
│   ├── 📁 static/                     # Static assets
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   │
│   └── 📁 utils/
│       └── init_data.py              # Database initialization
│
├── 📁 config/
│   └── config.py                     # Configuration classes
│
├── 📁 data/
│   ├── raw/                          # Raw uploaded files
│   ├── processed/                    # Processed CSV exports
│   └── uploads/                      # Temporary uploads
│
├── 📁 database/
│   └── dengue_app.db                 # SQLite database (auto-created)
│
├── 📁 migrations/                     # Database migrations (Flask-Migrate)
│
├── 📁 models/                         # Trained PyTorch models
│   └── dengue_stgnn_model.pth        # Your trained model (copy here)
│
├── 📁 tests/                          # Unit tests
│
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Full documentation (800 lines)
├── 📄 QUICK_START.md                  # Quick start guide (500 lines)
├── 📄 PROJECT_SUMMARY.md              # Project summary
└── 📄 ARCHITECTURE.md                 # This file
```

---

## 🔄 Data Flow Diagram

### 1. Dengue Case Data Flow

```
┌─────────────────┐
│  Admin/District │
│  Health User    │
└────────┬────────┘
         │ 1. Enter data via web form or upload CSV
         ↓
┌─────────────────────────────────────┐
│  DataPipelineService                │
│  - Validate data                    │
│  - Check duplicates                 │
│  - Save to database                 │
└────────┬────────────────────────────┘
         │ 2. Store in database
         ↓
┌─────────────────────────────────────┐
│  DengueCase Table                   │
│  - regency_id, year, month, cases   │
│  - reported_by_id, data_source      │
└────────┬────────────────────────────┘
         │ 3. Available for prediction
         ↓
┌─────────────────────────────────────┐
│  PredictionService                  │
│  - Use historical cases             │
│  - Generate predictions             │
└─────────────────────────────────────┘
```

### 2. Climate Data Flow

```
┌─────────────┐
│  Admin User │
└──────┬──────┘
       │ 1. Click "Fetch Climate"
       │    (Year, Month)
       ↓
┌────────────────────────────────────┐
│  DataPipelineService               │
│  fetch_all_climate_data()          │
└──────┬─────────────────────────────┘
       │ 2. For each regency:
       ↓
┌────────────────────────────────────┐
│  External API Call                 │
│  - NASA POWER (primary)            │
│  - OpenWeather (fallback)          │
│  - Daily data for entire month     │
└──────┬─────────────────────────────┘
       │ 3. Calculate monthly averages
       ↓
┌────────────────────────────────────┐
│  ClimateData Table                 │
│  - temp_min, temp_max, humidity    │
│  - precipitation, wind, cloud      │
└──────┬─────────────────────────────┘
       │ 4. Available for prediction
       ↓
┌────────────────────────────────────┐
│  PredictionService                 │
│  - Use climate features            │
└────────────────────────────────────┘
```

### 3. NDVI Data Flow

```
┌─────────────┐
│  Admin User │
└──────┬──────┘
       │ 1. Upload GeoTIFF file
       │    (Year, Month)
       ↓
┌────────────────────────────────────┐
│  DataPipelineService               │
│  process_ndvi_from_satellite()     │
└──────┬─────────────────────────────┘
       │ 2. Open GeoTIFF with rasterio
       ↓
┌────────────────────────────────────┐
│  Extract NDVI Values               │
│  - For each regency location       │
│  - Get pixel value at lat/lon      │
│  - Validate NDVI range (-1 to 1)   │
└──────┬─────────────────────────────┘
       │ 3. Save to database
       ↓
┌────────────────────────────────────┐
│  NDVIData Table                    │
│  - regency_id, year, month         │
│  - ndvi_value, is_imputed          │
└──────┬─────────────────────────────┘
       │ 4. If gaps exist:
       ↓
┌────────────────────────────────────┐
│  Imputation Service                │
│  - Forward fill / Backward fill    │
└──────┬─────────────────────────────┘
       │ 5. Available for prediction
       ↓
┌────────────────────────────────────┐
│  PredictionService                 │
│  - Use NDVI as feature             │
└────────────────────────────────────┘
```

### 4. Prediction Flow

```
┌─────────────────┐
│  Admin/District │
│  Health User    │
└────────┬────────┘
         │ 1. Request prediction (Year, Month)
         ↓
┌─────────────────────────────────────┐
│  PredictionService                  │
│  predict_single_regency()           │
└────────┬────────────────────────────┘
         │ 2. Prepare input data
         ↓
┌─────────────────────────────────────┐
│  Get Historical Data (4 months)     │
│  - Dengue cases                     │
│  - Climate data                     │
│  - NDVI data                        │
└────────┬────────────────────────────┘
         │ 3. Check data completeness
         ↓
┌─────────────────────────────────────┐
│  Load STGNN Model                   │
│  - From models/dengue_stgnn_model.pth│
│  - Using DenguePredictor            │
└────────┬────────────────────────────┘
         │ 4. Run prediction
         ↓
┌─────────────────────────────────────┐
│  Model Inference                    │
│  - Predicted cases                  │
│  - Zero probability                 │
└────────┬────────────────────────────┘
         │ 5. Calculate risk level
         ↓
┌─────────────────────────────────────┐
│  Risk Classification                │
│  - Low / Medium / High / Very High  │
│  - Based on thresholds              │
└────────┬────────────────────────────┘
         │ 6. Save to database
         ↓
┌─────────────────────────────────────┐
│  Prediction Table                   │
│  - predicted_cases, risk_level      │
│  - confidence bounds                │
└────────┬────────────────────────────┘
         │ 7. Generate recommendation
         ↓
┌─────────────────────────────────────┐
│  Health Recommendation              │
│  - Based on risk level              │
│  - Action plan for health office    │
└────────┬────────────────────────────┘
         │ 8. Display to user
         ↓
┌─────────────────────────────────────┐
│  Web Dashboard                      │
│  - Prediction with visualization    │
│  - Risk badge (color-coded)         │
│  - Recommendation text              │
└─────────────────────────────────────┘
```

---

## 🔐 Authentication Flow

```
┌──────────┐
│  Browser │
└────┬─────┘
     │ 1. POST /auth/login
     │    {username, password}
     ↓
┌─────────────────────────┐
│  AuthRoutes             │
│  - Validate credentials │
└────┬────────────────────┘
     │ 2. Check password
     ↓
┌─────────────────────────┐
│  User.check_password()  │
│  - Hash comparison      │
└────┬────────────────────┘
     │ 3. Create session
     ↓
┌─────────────────────────┐
│  Flask-Login            │
│  - login_user()         │
│  - Set session cookie   │
└────┬────────────────────┘
     │ 4. Redirect based on role
     ↓
┌─────────────────────────┐
│  Role-based redirect:   │
│  - Admin → /admin/dashboard         │
│  - District Health → /district/dashboard│
│  - Public → /dashboard  │
└─────────────────────────┘
```

### Protected Route Flow

```
┌──────────┐
│  Request │
└────┬─────┘
     │ GET /admin/data-management
     ↓
┌─────────────────────────┐
│  @admin_required        │
│  decorator              │
└────┬────────────────────┘
     │ Check authentication
     ↓
┌─────────────────────────┐
│  current_user           │
│  .is_authenticated?     │
└────┬────────────────────┘
     │ YES → Check role
     ↓
┌─────────────────────────┐
│  current_user           │
│  .is_admin()?           │
└────┬────────────────────┘
     │ YES → Allow access
     │ NO  → Redirect with error
     ↓
┌─────────────────────────┐
│  Execute route function │
│  Return response        │
└─────────────────────────┘
```

---

## 🗄️ Database Schema

```
┌─────────────────┐         ┌──────────────────┐
│     User        │         │     Regency      │
├─────────────────┤         ├──────────────────┤
│ id (PK)         │         │ id (PK)          │
│ username        │         │ name             │
│ email           │         │ latitude         │
│ password_hash   │         │ longitude        │
│ role            │         │ population       │
│ regency         │         │ area_km2         │
│ is_active       │         │ is_active        │
│ created_at      │         └──────┬───────────┘
│ last_login      │                │
└────────┬────────┘                │
         │                         │
         │                         │
         │ reported_by_id (FK)     │ regency_id (FK)
         ↓                         ↓
┌─────────────────────────────────────────┐
│            DengueCase                   │
├─────────────────────────────────────────┤
│ id (PK)                                 │
│ regency_id (FK → Regency)               │
│ year, month, week                       │
│ cases                                   │
│ data_source                             │
│ reported_by_id (FK → User)              │
│ created_at, updated_at                  │
└─────────────────────────────────────────┘

         ┌────── Regency.id ──────┐
         ↓                        ↓
┌─────────────────────┐  ┌─────────────────────┐
│   ClimateData       │  │    NDVIData         │
├─────────────────────┤  ├─────────────────────┤
│ id (PK)             │  │ id (PK)             │
│ regency_id (FK)     │  │ regency_id (FK)     │
│ year, month         │  │ year, month         │
│ temperature_*       │  │ ndvi_value          │
│ humidity            │  │ data_source         │
│ precipitation_*     │  │ is_imputed          │
│ pressure            │  │ processing_date     │
│ wind_*              │  └─────────────────────┘
│ cloud_cover         │
│ data_source         │
│ fetched_at          │
└─────────────────────┘

         ┌────── Regency.id ──────┐
         ↓                        
┌─────────────────────────────────┐
│         Prediction              │
├─────────────────────────────────┤
│ id (PK)                         │
│ regency_id (FK → Regency)       │
│ year, month                     │
│ predicted_cases                 │
│ zero_probability                │
│ confidence_lower                │
│ confidence_upper                │
│ actual_cases                    │
│ model_version                   │
│ risk_level                      │
│ prediction_date                 │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│       ModelVersion              │
├─────────────────────────────────┤
│ id (PK)                         │
│ version_name                    │
│ model_file                      │
│ description                     │
│ mae, rmse, r2_score             │
│ training_data_file              │
│ training_date                   │
│ training_samples                │
│ is_active                       │
│ created_at                      │
└─────────────────────────────────┘

         ┌────── User.id ──────┐
         ↓                     
┌─────────────────────────────────┐
│    DataProcessingLog            │
├─────────────────────────────────┤
│ id (PK)                         │
│ user_id (FK → User)             │
│ process_type                    │
│ status                          │
│ records_processed               │
│ error_message                   │
│ details (JSON)                  │
│ started_at                      │
│ completed_at                    │
└─────────────────────────────────┘
```

---

## 🌐 URL Routing Map

### Public Routes (No auth required)
```
GET  /                          → Landing page
GET  /dashboard                 → Public dashboard (all regencies)
GET  /regency/<id>              → Regency detail page
GET  /statistics                → Provincial statistics
GET  /about                     → About page
GET  /api/regencies             → API: List regencies (JSON)
GET  /api/cases/<id>            → API: Dengue cases (JSON)
GET  /api/predictions/<id>      → API: Predictions (JSON)
```

### Auth Routes
```
GET  /auth/login                → Login page
POST /auth/login                → Process login
GET  /auth/logout               → Logout
GET  /auth/register             → Registration (disabled)
```

### Admin Routes (Admin only)
```
GET  /admin/dashboard           → Admin dashboard
GET  /admin/data-management     → Data management interface
POST /admin/data/dengue/add     → Add dengue case
POST /admin/data/dengue/bulk-import → Bulk import CSV
POST /admin/data/climate/fetch  → Fetch climate data (API)
POST /admin/data/ndvi/upload    → Upload NDVI GeoTIFF
POST /admin/data/ndvi/impute    → Impute missing NDVI
POST /admin/data/export         → Export to CSV
POST /admin/predictions/generate → Generate predictions
POST /admin/model/upload        → Upload model file
POST /admin/model/activate/<id> → Activate model
GET  /admin/logs                → View processing logs
```

### District Health Routes (District health + Admin)
```
GET  /district/dashboard        → District dashboard
GET  /district/update-cases     → Case update form
POST /district/cases/add        → Add/update cases
GET  /district/predictions      → View predictions
POST /district/request-prediction → Request prediction
GET  /district/reports          → Reports & statistics
```

---

## 🔧 Technology Stack Details

### Backend
- **Flask 3.0:** Web framework
- **SQLAlchemy 2.0:** ORM for database
- **Flask-Login:** Session management
- **Flask-Migrate:** Database migrations
- **Werkzeug:** Password hashing, security

### Frontend
- **Bootstrap 5.3:** Responsive UI
- **Chart.js 4.4:** Data visualization
- **Font Awesome 6.4:** Icons
- **jQuery 3.7:** DOM manipulation (optional)

### Data Processing
- **Pandas 2.1:** Data manipulation
- **NumPy 1.26:** Numerical operations
- **Rasterio 1.3:** GeoTIFF processing

### Machine Learning
- **PyTorch 2.1:** Model inference
- **Scikit-learn 1.3:** Data preprocessing
- **Custom STGNN:** Your dengue prediction model

### External APIs
- **NASA POWER:** Climate data (free)
- **OpenWeather:** Backup climate (requires key)

### Development
- **pytest:** Unit testing
- **Flask-WTF:** Form handling
- **python-dotenv:** Environment variables

### Production
- **Gunicorn:** WSGI server
- **PostgreSQL:** Production database
- **Nginx:** Reverse proxy (recommended)

---

## 🚀 Deployment Architecture (Production)

```
                    Internet
                       │
                       ↓
            ┌──────────────────┐
            │   Nginx (80/443) │ ← Reverse Proxy + SSL
            │   - Static files │
            │   - Load balance │
            └────────┬─────────┘
                     │
                     ↓
        ┌────────────────────────────┐
        │  Gunicorn (8000-8003)      │ ← WSGI Server
        │  - 4 worker processes      │
        │  - Flask app instances     │
        └────────┬───────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ↓                 ↓
┌───────────────┐  ┌──────────────┐
│  PostgreSQL   │  │  File System │
│  Database     │  │  - Uploads   │
│  - Persistent │  │  - Models    │
│  - Backup     │  │  - Logs      │
└───────────────┘  └──────────────┘
```

---

## 📊 Performance Characteristics

### Response Times (Expected)
- Login: < 100ms
- Dashboard load: < 500ms
- Add dengue case: < 200ms
- Climate fetch (all regencies): ~2-5 minutes
- NDVI upload: ~1-2 minutes
- Prediction generation: < 5 seconds
- Export CSV: < 3 seconds

### Scalability
- Concurrent users: 50+ (with Gunicorn)
- Database: Supports 100,000+ records
- Model inference: <1 second per prediction
- API rate limits: Handled by caching

### Storage Requirements
- Database: ~10 MB per year of data
- Model files: ~5-50 MB per model
- Uploads: Temporary (cleaned)
- Logs: ~1 MB per month

---

## 🔒 Security Layers

```
Layer 1: Network
├─ HTTPS (SSL/TLS)
├─ Firewall rules
└─ Rate limiting

Layer 2: Application
├─ Session management (Flask-Login)
├─ CSRF tokens
├─ Input validation
└─ SQL injection prevention (ORM)

Layer 3: Authentication
├─ Password hashing (Werkzeug)
├─ Salted passwords
└─ Session expiry

Layer 4: Authorization
├─ Role-based access (decorators)
├─ Regency-level permissions
└─ Audit logging

Layer 5: Data
├─ Database backups
├─ Data encryption (optional)
└─ Access logs
```

---

## 📈 Monitoring & Logging

### Application Logs
```
DataProcessingLog table:
- User actions
- Process status
- Error messages
- Timestamps
```

### System Logs
```
Flask logs:
- HTTP requests
- Errors/warnings
- Performance metrics
```

### Metrics to Monitor
- User login frequency
- Data update frequency
- Prediction accuracy
- API call success rate
- System uptime

---

## 🎯 System Capabilities Summary

| Capability | Status | Notes |
|-----------|--------|-------|
| Multi-user auth | ✅ | 3 roles: admin, district_health, public |
| Data input | ✅ | Manual + bulk CSV |
| Climate data | ✅ | Auto-fetch from APIs |
| NDVI processing | ✅ | Upload + auto-extract |
| Data export | ✅ | CSV format |
| Prediction | ✅ | STGNN model integration |
| Risk classification | ✅ | 4 levels |
| Recommendations | ✅ | Auto-generated |
| Public dashboard | ✅ | Read-only access |
| Audit trail | ✅ | All operations logged |
| Role permissions | ✅ | Enforced at route level |
| API endpoints | ✅ | JSON responses |
| Responsive UI | ✅ | Mobile-friendly |
| Production ready | ✅ | Scalable architecture |

---

**This architecture provides a solid foundation for a production dengue prediction system!**
