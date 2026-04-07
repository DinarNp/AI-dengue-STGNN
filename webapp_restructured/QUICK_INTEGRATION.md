# 🚀 Quick Integration Guide

## ⚡ Fast Start (5 Minutes)

### Step 1: Import Your Data (2 minutes)
```bash
cd webapp_restructured
python scripts/import_existing_data.py
```

### Step 2: Start Application (30 seconds)
```bash
python run.py
```

### Step 3: Train Model (Open browser)
```
1. Go to: http://localhost:5000
2. Login: admin / admin123
3. Go to: Admin → Training
4. Click "Export Data" (2021-2024)
5. Click "Start Training" (wait 5-20 min)
6. Click "Activate" when done
```

### Step 4: Generate Predictions
```
1. Go to: Admin → Predictions
2. Select year/month
3. Click "Generate"
4. View in dashboards!
```

---

## 📋 What Was Added

### 1. Import Script
**Location:** `scripts/import_existing_data.py`

**What it does:**
- Reads your CSV files from `data/fix/`
- Imports dengue cases, climate data, NDVI data
- Stores in database tables
- Handles 2021-2024 and 2025 data

**Usage:**
```bash
python scripts/import_existing_data.py
```

### 2. Training Service
**Location:** `app/services/training.py`

**Features:**
- Export data from database to CSV
- Train STGNN model using your existing pipeline
- Save trained models with metrics
- Activate/deactivate models
- Track training logs

### 3. Admin Training Interface
**Location:** `app/templates/admin/training.html`

**Features:**
- Visual training interface
- One-click data export
- One-click model training
- Model management (activate/delete)
- Training progress tracking
- Performance metrics display

### 4. New Admin Routes
**Added to:** `app/routes/admin.py`

- `/admin/training` - Training dashboard
- `/admin/training/export-data` - Export training data
- `/admin/training/train-model` - Train new model
- `/admin/training/activate-model/<id>` - Activate model
- `/admin/training/delete-model/<id>` - Delete model

---

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────┐
│  YOUR EXISTING DATA                             │
├─────────────────────────────────────────────────┤
│  data/fix/data_monthly_5kab_2021_2024_ndvi.csv │
│  data/fix/data_monthly_5kab_2025_ndvi.csv      │
│  NEO/MOD_NDVI_16/*.TIFF                         │
└──────────────────┬──────────────────────────────┘
                   ↓
         ┌─────────────────────┐
         │   IMPORT SCRIPT     │
         │ (One-time setup)    │
         └──────────┬──────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│  WEB APPLICATION DATABASE                        │
├──────────────────────────────────────────────────┤
│  DengueCase Table (300+ records)                 │
│  ClimateData Table (300+ records)                │
│  NDVIData Table (300+ records)                   │
└──────────────────┬───────────────────────────────┘
                   ↓
         ┌─────────────────────┐
         │  ADMIN INTERFACE    │
         │  → Training Page    │
         └──────────┬──────────┘
                   ↓
         ┌─────────────────────┐
         │  EXPORT DATA        │
         │  (2021-2024)        │
         └──────────┬──────────┘
                   ↓
         ┌─────────────────────┐
         │  TRAIN MODEL        │
         │  (5-20 minutes)     │
         └──────────┬──────────┘
                   ↓
         ┌─────────────────────┐
         │  TRAINED MODEL      │
         │  (.pth file)        │
         └──────────┬──────────┘
                   ↓
         ┌─────────────────────┐
         │  ACTIVATE MODEL     │
         └──────────┬──────────┘
                   ↓
         ┌─────────────────────┐
         │  GENERATE           │
         │  PREDICTIONS        │
         └──────────┬──────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│  VIEW IN DASHBOARDS                              │
├──────────────────────────────────────────────────┤
│  - Public Dashboard (all regencies)              │
│  - Admin Dashboard (manage & predict)            │
│  - District Health Dashboard (their regency)     │
└──────────────────────────────────────────────────┘
```

---

## 📊 Data Mapping

### Your CSV → Database

| CSV Column | Database Table | Database Column |
|------------|---------------|-----------------|
| Year | DengueCase | year |
| Region | DengueCase | regency_id (mapped) |
| Month | DengueCase | month |
| Cases | DengueCase | cases |
| Latitude | Regency | latitude |
| Longitude | Regency | longitude |
| NDVI | NDVIData | ndvi_value |
| Cloud_Cover | ClimateData | cloud_cover |
| Humidity | ClimateData | humidity |
| Precipitation_Total | ClimateData | precipitation_total |
| Temperature_Min | ClimateData | temperature_min |
| Temperature_Max | ClimateData | temperature_max |
| Temperature_Avg | ClimateData | temperature_avg |
| Pressure | ClimateData | pressure |
| Wind_Speed | ClimateData | wind_speed |
| Wind_Direction | ClimateData | wind_direction |

---

## 🎯 Key Features

### 1. Seamless Integration
- Your existing CSV data → Database
- Your existing STGNN model → Web interface
- No code changes needed to original model

### 2. Web-Based Training
- Visual interface instead of command line
- Progress tracking
- Error handling
- Model versioning

### 3. Automatic Predictions
- Train once, predict many times
- Use in public dashboard
- Use in district health dashboards
- API access available

---

## ⚙️ Configuration

### Import Script Configuration

Edit `scripts/import_existing_data.py` if you have different file paths:

```python
csv_files = [
    {
        'path': '../data/fix/data_monthly_5kab_2021_2024_ndvi.csv',
        'source': 'historical_2021_2024'
    },
    {
        'path': '../data/fix/data_monthly_5kab_2025_ndvi.csv',
        'source': 'current_2025'
    }
]
```

### Training Configuration

Model config is inherited from your existing `config/config.py`:
- WINDOW_SIZE_MONTHLY = 4
- HIDDEN_DIM = 256
- NUM_LAYERS = 4
- EPOCHS = 1000
- etc.

---

## 🔍 Verify Everything Works

### 1. Check Data Import
```bash
cd webapp_restructured
python -c "
from app import create_app
from app.models import db, DengueCase, ClimateData, NDVIData
app = create_app()
with app.app_context():
    print(f'Dengue Cases: {DengueCase.query.count()}')
    print(f'Climate Data: {ClimateData.query.count()}')
    print(f'NDVI Data: {NDVIData.query.count()}')
"
```

Expected output:
```
Dengue Cases: 300
Climate Data: 300
NDVI Data: 300
```

### 2. Check Web Interface
```bash
python run.py
```

Visit: http://localhost:5000/admin/training

Should show:
- Training Status card
- Quick Training section
- Trained Models table (empty at first)

### 3. Test Training
1. Click "Export Data" (2021-2024)
2. Click "Start Training"
3. Wait for completion
4. Check metrics (MAE, RMSE, R²)

---

## 💡 Pro Tips

### Tip 1: Keep Original Data
Your CSV files are not modified. They're only read and imported to database.

### Tip 2: Re-import Anytime
You can run the import script multiple times. It will update existing records.

### Tip 3: Multiple Models
Train different models with different data ranges:
- Model_2021_2023 (training set)
- Model_2021_2024 (full historical)
- Compare performance!

### Tip 4: Export Predictions
After generating predictions, export via Admin → Data Management → Export

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Import script: "No regencies found" | Run `python run.py` first to initialize database |
| Training: "Modules not available" | Check parent directory has config/, experiments/, models/ |
| Training takes forever | Normal! 5-20 minutes depending on data size |
| Predictions not showing | Activate model first via Training page |
| Dashboard empty | Generate predictions first |

---

## 📞 Quick Commands Reference

```bash
# Import data (one-time)
python scripts/import_existing_data.py

# Start app
python run.py

# Check database
sqlite3 database/dengue_app.db "SELECT COUNT(*) FROM dengue_cases;"

# View logs
tail -f nohup.out
```

---

## ✅ Success Checklist

- [ ] Import script completed successfully
- [ ] Web app starts without errors
- [ ] Can login as admin
- [ ] Training page loads
- [ ] Data export works
- [ ] Model training completes
- [ ] Model appears in list
- [ ] Model activated
- [ ] Predictions generated
- [ ] Dashboards show data

---

## 🎉 You're Done!

You now have:
- ✅ All your data in the database
- ✅ Web interface for training
- ✅ Trained STGNN model
- ✅ Prediction system working
- ✅ Dashboards populated

**Next:** Share with health offices and start predicting! 🦟📊

For detailed information, see:
- DATA_INTEGRATION_GUIDE.md (comprehensive guide)
- README.md (full documentation)
- QUICK_START.md (application usage)
