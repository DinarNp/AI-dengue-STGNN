# Data Integration & Model Training Guide

## 🎯 Overview

This guide explains how to:
1. Import your existing CSV datasets into the web application
2. Use the new model training functionality
3. Make predictions with trained models

---

## 📊 Step 1: Import Existing Data

### Your Current Data Files

You have these datasets:
- `data/fix/data_monthly_5kab_2021_2024_ndvi.csv` (Historical: 2021-2024)
- `data/fix/data_monthly_5kab_2025_ndvi.csv` (Current: 2025)
- NEO NDVI satellite data in `NEO/MOD_NDVI_16/` and `NEO/MOD_NDVI_16_2025_2026/`

### Import Script

We've created an import script that will migrate all your CSV data into the web application database.

**Run the import script:**

```bash
cd webapp_restructured
python scripts/import_existing_data.py
```

### What the Script Does

1. **Reads your CSV files**
2. **Maps data to database tables:**
   - Dengue cases → `DengueCase` table
   - Climate data → `ClimateData` table
   - NDVI values → `NDVIData` table
3. **Links data to regencies**
4. **Handles duplicates** (updates existing records)

### Expected Output

```
============================================================
DATA IMPORT UTILITY
Importing existing CSV data to database
============================================================

Found 5 regencies in database

============================================================
Importing data from: ../data/fix/data_monthly_5kab_2021_2024_ndvi.csv
============================================================
Loaded 240 rows from CSV
Found 5 regencies in database
Processed 50 rows...
Processed 100 rows...
Processed 150 rows...
Processed 200 rows...

============================================================
Import Summary:
============================================================
Dengue Cases:  240 imported, 0 updated
Climate Data:  240 imported, 0 updated
NDVI Data:     240 imported, 0 updated
Skipped:       0 rows
============================================================

... (continues for 2025 data)

============================================================
IMPORT COMPLETE!
Imported data from 2 CSV files
============================================================

Database Statistics:
  Dengue Cases: 300
  Climate Data: 300
  NDVI Data:    300
```

---

## 🧠 Step 2: Train Your Model

### Option A: Train via Web Interface (Recommended)

1. **Start the application:**
   ```bash
   python run.py
   ```

2. **Login as admin:**
   - Username: `admin`
   - Password: `admin123`

3. **Navigate to Admin → Model Training**
   ```
   http://localhost:5000/admin/training
   ```

4. **Export Training Data:**
   - Year Range: 2021 to 2024
   - Click "Export Data"
   - This creates: `training_data_2021_2024.csv`

5. **Train Model:**
   - Model Name: `STGNN_v1.0`
   - Data File: `training_data_2021_2024.csv`
   - Click "Start Training"
   - Wait 5-20 minutes (depends on your machine)

6. **Activate Model:**
   - Once training completes, click "Activate" button
   - This model will now be used for all predictions

### Option B: Train via Command Line

You can still use your existing training scripts:

```bash
# In parent directory
python main2.py  # For monthly data training
```

Then upload the trained model via web interface:
- Admin → Model Training → Upload Model

---

## 🔮 Step 3: Generate Predictions

### Via Web Interface

1. **Login as admin**

2. **Navigate to Admin → Data Management**

3. **Click "Generate Predictions"**
   - Year: 2025
   - Month: (select month)
   - Click "Generate for All Regencies"

4. **View predictions:**
   - Admin Dashboard
   - Public Dashboard
   - District Health Dashboard (for their regency)

### Via API

```python
import requests

# Generate prediction
response = requests.post('http://localhost:5000/admin/predictions/generate', 
    json={'year': 2025, 'month': 5},
    auth=('admin', 'admin123')
)

result = response.json()
print(result)
```

---

## 📁 Data Structure in Database

### Tables Created

```
┌─────────────────┐
│   DengueCase    │  ← Your dengue case data
├─────────────────┤
│ regency_id      │
│ year            │
│ month           │
│ cases           │
│ data_source     │  'historical_2021_2024' or 'current_2025'
└─────────────────┘

┌─────────────────┐
│  ClimateData    │  ← Your weather data
├─────────────────┤
│ regency_id      │
│ year, month     │
│ temperature_*   │
│ humidity        │
│ precipitation   │
│ wind_*, etc     │
└─────────────────┘

┌─────────────────┐
│    NDVIData     │  ← Your satellite data
├─────────────────┤
│ regency_id      │
│ year, month     │
│ ndvi_value      │
│ is_imputed      │
└─────────────────┘

┌─────────────────┐
│  Prediction     │  ← Model predictions
├─────────────────┤
│ regency_id      │
│ year, month     │
│ predicted_cases │
│ risk_level      │
└─────────────────┘
```

---

## 🔄 Data Flow

### Complete Workflow

```
Your CSV Files
    ↓
Import Script (scripts/import_existing_data.py)
    ↓
Database Tables (DengueCase, ClimateData, NDVIData)
    ↓
Export Training Data (Admin → Training → Export)
    ↓
training_data_YYYY_YYYY.csv
    ↓
Train Model (Admin → Training → Train)
    ↓
Trained Model (.pth file)
    ↓
Activate Model
    ↓
Generate Predictions (Admin → Predictions)
    ↓
View in Dashboards (Public/Admin/District Health)
```

---

## 🎓 Detailed Training Example

### Full Step-by-Step

```bash
# 1. Import your existing data
cd webapp_restructured
python scripts/import_existing_data.py

# 2. Start the web app
python run.py

# 3. Open browser
http://localhost:5000

# 4. Login as admin
Username: admin
Password: admin123

# 5. Go to Admin → Training
# 6. Export data (2021-2024)
# 7. Train model
# 8. Wait for training (5-20 min)
# 9. Activate model
# 10. Generate predictions!
```

---

## 📊 Verify Data Import

### Check Database Contents

```python
# In Python shell
from webapp_restructured.app import create_app
from webapp_restructured.app.models import db, DengueCase, ClimateData, NDVIData

app = create_app('development')
with app.app_context():
    print(f"Dengue Cases: {DengueCase.query.count()}")
    print(f"Climate Data: {ClimateData.query.count()}")
    print(f"NDVI Data: {NDVIData.query.count()}")
    
    # Show sample data
    sample = DengueCase.query.first()
    print(f"\nSample: {sample.regency.name}, {sample.year}-{sample.month}, Cases: {sample.cases}")
```

### Via Web Interface

1. Login as admin
2. Go to Admin → Data Management
3. Check "Data Completeness Overview" table
4. Should show data for all regencies

---

## 🔧 Troubleshooting

### Import Script Issues

**Problem:** "No regencies found in database"
**Solution:** Run the app first to initialize default data
```bash
python run.py
# Let it start, then Ctrl+C
# Then run import script
python scripts/import_existing_data.py
```

**Problem:** "File not found"
**Solution:** Check that CSV files exist in correct locations
```bash
ls -la ../data/fix/data_monthly_5kab_2021_2024_ndvi.csv
ls -la ../data/fix/data_monthly_5kab_2025_ndvi.csv
```

### Training Issues

**Problem:** "Training modules not available"
**Solution:** Ensure parent directory code is accessible
```bash
# Check if these exist
ls -la ../config/config.py
ls -la ../experiments/dengue_pipeline.py
ls -la ../models/stgnn.py
```

**Problem:** Training takes too long
**Solution:** 
- Expected: 5-20 minutes for 240 records
- Use CPU if GPU not available
- Check system resources

**Problem:** Out of memory
**Solution:**
- Reduce batch size in config
- Use smaller data range
- Close other applications

---

## 📈 Using Predictions

### View Predictions

**Public Dashboard:**
- http://localhost:5000/dashboard
- Shows all regencies
- No login required

**Admin Dashboard:**
- http://localhost:5000/admin/dashboard
- Full control
- Generate new predictions

**District Health:**
- http://localhost:5000/district/dashboard
- Their regency only
- View predictions and recommendations

### Export Predictions

```python
# Via API
import requests
import pandas as pd

# Get predictions for all regencies
predictions = []
for regency_id in [1, 2, 3, 4, 5]:
    resp = requests.get(f'http://localhost:5000/api/predictions/{regency_id}')
    predictions.extend(resp.json())

# Save to CSV
df = pd.DataFrame(predictions)
df.to_csv('predictions_export.csv', index=False)
```

---

## 🎯 Best Practices

### Data Management

1. **Import historical data first** (2021-2024)
2. **Verify import** via web interface
3. **Train initial model** with historical data
4. **Import current data** (2025)
5. **Generate predictions** for future months

### Model Training

1. **Use consistent data ranges** (e.g., full years)
2. **Train on historical data only** (not future data)
3. **Activate best performing model**
4. **Keep multiple model versions** for comparison
5. **Retrain periodically** with new data

### Predictions

1. **Ensure 4 months of historical data** before predicting
2. **Update data monthly** for best accuracy
3. **Review predictions** before sharing
4. **Compare with actual cases** to validate

---

## 📝 Summary

You now have:
- ✅ Import script for your existing CSV data
- ✅ Web interface for model training
- ✅ Integrated prediction system
- ✅ Data stored in proper database structure
- ✅ Easy workflow for future updates

**Next Steps:**
1. Run import script
2. Train your first model
3. Generate predictions
4. Share with health offices!

---

## 🆘 Need Help?

Check:
1. This guide
2. QUICK_START.md
3. README.md
4. Application logs in database
5. Console output during training

**Common Workflow:**
```bash
# One-time setup
python scripts/import_existing_data.py

# Regular use
python run.py
# → Login → Train models → Generate predictions → View dashboards
```

Good luck with your dengue prediction system! 🦟📊🎯
