# 🎓 Complete Training & Prediction Guide

## 🎯 Your Requirements

You want to:
1. ✅ **Train STGNN model** via admin interface
2. ✅ **Use trained model** for predictions in admin & public dashboards
3. ✅ **Store data in separate CSV files** (dengue.csv, climate.csv, ndvi.csv)
4. ✅ **Edit/CRUD data** easily via admin interface
5. ✅ **Clear pipeline** from data management → training → prediction

---

## 📊 Recommended Data Structure

### Option A: Separate CSV Files (Easier for Editing) ⭐ RECOMMENDED

Store 3 separate CSV files that are easy to edit:

```
webapp_restructured/data/master/
├── dengue_cases.csv          ← Dengue case data
├── climate_data.csv          ← Weather data
└── ndvi_data.csv             ← Satellite NDVI data
```

**Benefits:**
- ✅ Easy to edit in Excel/Google Sheets
- ✅ Clear separation of data types
- ✅ Easy to update individual datasets
- ✅ Can be version controlled
- ✅ Simple to backup

**Format for each file:**

**dengue_cases.csv:**
```csv
Year,Month,Region,Cases,Notes
2021,1,KAB BANTUL,60,Verified data
2021,2,KAB BANTUL,29,
2021,3,KAB BANTUL,51,
```

**climate_data.csv:**
```csv
Year,Month,Region,Temperature_Min,Temperature_Max,Temperature_Avg,Humidity,Precipitation_Total,Pressure,Wind_Speed,Wind_Direction,Cloud_Cover
2021,1,KAB BANTUL,24.5,30.2,29.7,66.75,28.21,1007.75,4.16,235.5,90.0
2021,2,KAB BANTUL,24.2,31.0,30.6,64.0,35.18,1009.0,4.30,213.5,73.0
```

**ndvi_data.csv:**
```csv
Year,Month,Region,NDVI,Source,Is_Imputed
2021,1,KAB BANTUL,0.0209,MODIS,No
2021,2,KAB BANTUL,0.0213,MODIS,No
2021,3,KAB BANTUL,0.0217,MODIS,No
```

---

## 🔄 Complete Pipeline: Data → Training → Prediction

### Visual Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: DATA MANAGEMENT (Admin Interface)             │
├─────────────────────────────────────────────────────────┤
│  CSV Files (Editable)                                   │
│    ├── dengue_cases.csv                                 │
│    ├── climate_data.csv                                 │
│    └── ndvi_data.csv                                    │
│                                                          │
│  Admin Actions:                                          │
│    • Upload CSV                                          │
│    • Edit via web form                                   │
│    • Download CSV                                        │
│    • View/Delete records                                 │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: DATA VALIDATION & MERGE                        │
├─────────────────────────────────────────────────────────┤
│  System automatically:                                   │
│    • Validates data completeness                         │
│    • Checks for missing months                           │
│    • Merges 3 CSV files into training format            │
│    • Creates: training_data_merged.csv                  │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: MODEL TRAINING (Admin Interface)              │
├─────────────────────────────────────────────────────────┤
│  Admin clicks "Train Model"                             │
│    ↓                                                     │
│  System:                                                 │
│    • Reads training_data_merged.csv                     │
│    • Trains STGNN model (5-20 minutes)                  │
│    • Saves model: models/STGNN_v1.pth                   │
│    • Records metrics (MAE, RMSE, R²)                    │
│    • Stores in database                                  │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: MODEL ACTIVATION                               │
├─────────────────────────────────────────────────────────┤
│  Admin selects best model and clicks "Activate"         │
│    ↓                                                     │
│  System:                                                 │
│    • Marks model as active                              │
│    • Loads model for predictions                        │
│    • Ready for use!                                     │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 5: GENERATE PREDICTIONS (Admin Interface)        │
├─────────────────────────────────────────────────────────┤
│  Admin generates predictions:                           │
│    • Select year/month                                   │
│    • Click "Generate Predictions"                       │
│    • System uses active model                           │
│    • Saves predictions to database                      │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 6: VIEW PREDICTIONS (All Users)                  │
├─────────────────────────────────────────────────────────┤
│  Public Dashboard:                                       │
│    • View all regencies predictions                     │
│    • See risk levels                                    │
│    • View trends                                        │
│                                                          │
│  District Health Dashboard:                             │
│    • View their regency predictions                     │
│    • Get recommendations                                │
│    • Update cases                                       │
│                                                          │
│  Admin Dashboard:                                       │
│    • View all predictions                               │
│    • Compare with actual cases                          │
│    • Generate new predictions                           │
└─────────────────────────────────────────────────────────┘
```

---

## 💻 Implementation Plan

I'll create a complete system with:

### 1. Data Management Interface
- Upload/Download CSV files
- Edit data via web forms
- CRUD operations (Create, Read, Update, Delete)
- Data validation
- Automatic merging for training

### 2. Training Interface
- One-click training from CSV files
- Progress tracking
- Model versioning
- Performance metrics
- Model activation

### 3. Prediction Interface
- Generate predictions for specific months
- Batch prediction (multiple months)
- View prediction results
- Export predictions

### 4. Dashboard Integration
- Public: View all predictions
- District: View their predictions + recommendations
- Admin: Manage everything

---

## 📁 File Structure (Simplified)

```
webapp_restructured/
│
├── data/
│   ├── master/                      ← Your editable CSV files ⭐
│   │   ├── dengue_cases.csv
│   │   ├── climate_data.csv
│   │   └── ndvi_data.csv
│   │
│   ├── training/                    ← Generated training files
│   │   └── training_data_merged.csv
│   │
│   └── exports/                     ← Downloaded files
│
├── models/                          ← Trained model files
│   ├── STGNN_v1.0_20250407.pth
│   ├── STGNN_v2.0_20250410.pth
│   └── active_model.pth            ← Symlink to active model
│
└── predictions/                     ← Prediction results
    └── predictions_2025.csv
```

---

## 🎯 Step-by-Step Usage

### STEP 1: Prepare Your Data

**Option A: Start with your existing data**
```bash
# Copy your existing CSV to master folder
cp data/fix/data_monthly_5kab_2021_2024_ndvi.csv \
   webapp_restructured/data/master/dengue_cases.csv
```

**Option B: Create from template**
1. Login as admin
2. Go to: Admin → Data Management → Download Templates
3. Fill in Excel/Google Sheets
4. Upload back to system

### STEP 2: Upload/Edit Data

**Via Web Interface:**
1. Login as admin
2. Go to: Admin → Data Management
3. Choose action:
   - **Upload CSV** → Select file → Upload
   - **Edit Record** → Find row → Click Edit → Save
   - **Add Record** → Click Add → Fill form → Save
   - **Delete Record** → Find row → Click Delete → Confirm

### STEP 3: Train Model

**Via Admin Interface:**
1. Go to: Admin → Model Training
2. Click "Prepare Training Data" 
   - System merges 3 CSV files
   - Shows data summary
3. Click "Train New Model"
   - Enter model name: "STGNN_2025_Jan"
   - Select date range: 2021-2024
   - Click "Start Training"
4. Wait 5-20 minutes
5. View metrics (MAE, RMSE, R²)
6. Click "Activate" to use this model

### STEP 4: Generate Predictions

**Via Admin Interface:**
1. Go to: Admin → Predictions
2. Select:
   - Year: 2025
   - Month: 5 (May)
   - Regencies: All (or select specific)
3. Click "Generate Predictions"
4. View results immediately

### STEP 5: View in Dashboards

**Public Dashboard:**
- Visit: http://localhost:5000/dashboard
- See all regencies with predictions
- Color-coded risk levels

**District Health Dashboard:**
- Login as district user
- See their regency prediction
- Get health recommendations

---

## 📝 Admin Interface Features

### Data Management Page

```
┌────────────────────────────────────────────────┐
│  Data Management                               │
├────────────────────────────────────────────────┤
│                                                │
│  [Dengue Cases] [Climate Data] [NDVI Data]    │
│                                                │
│  Current File: dengue_cases.csv (240 records) │
│                                                │
│  Actions:                                      │
│  [Upload New CSV] [Download Template]         │
│  [Download Current Data] [Validate Data]      │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │ Year | Month | Region      | Cases      │ │
│  ├──────────────────────────────────────────┤ │
│  │ 2021 | 1     | KAB BANTUL  | 60  [Edit] │ │
│  │ 2021 | 2     | KAB BANTUL  | 29  [Edit] │ │
│  │ 2021 | 3     | KAB BANTUL  | 51  [Edit] │ │
│  │ ...                                      │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  [Add New Record] [Bulk Import] [Export All]  │
└────────────────────────────────────────────────┘
```

### Training Page

```
┌────────────────────────────────────────────────┐
│  Model Training                                │
├────────────────────────────────────────────────┤
│                                                │
│  Step 1: Data Preparation                     │
│  ┌──────────────────────────────────────────┐ │
│  │ Dengue Cases:  240 records ✓             │ │
│  │ Climate Data:  240 records ✓             │ │
│  │ NDVI Data:     240 records ✓             │ │
│  │ Date Range:    2021-01 to 2024-12        │ │
│  │ Completeness:  100%                      │ │
│  └──────────────────────────────────────────┘ │
│  [Prepare Training Data]                       │
│                                                │
│  Step 2: Train Model                          │
│  Model Name: [STGNN_v1.0          ]           │
│  Date Range: [2021] to [2024]                 │
│  [Start Training]                              │
│                                                │
│  Step 3: Trained Models                       │
│  ┌──────────────────────────────────────────┐ │
│  │ Model       | Date    | MAE   | Status  │ │
│  ├──────────────────────────────────────────┤ │
│  │ STGNN_v1.0  | Jan 7   | 0.85  | Active  │ │
│  │ STGNN_v0.9  | Jan 5   | 0.92  | [Activate]│
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

### Prediction Page

```
┌────────────────────────────────────────────────┐
│  Generate Predictions                          │
├────────────────────────────────────────────────┤
│                                                │
│  Active Model: STGNN_v1.0 (MAE: 0.85)         │
│                                                │
│  Prediction Settings:                         │
│  Year:     [2025 ▼]                           │
│  Month:    [May  ▼]                           │
│  Regency:  [☑ All Regencies]                  │
│                                                │
│  [Generate Predictions]                        │
│                                                │
│  Recent Predictions:                          │
│  ┌──────────────────────────────────────────┐ │
│  │ Date    | Regency      | Pred | Risk    │ │
│  ├──────────────────────────────────────────┤ │
│  │ 2025-05 | KAB BANTUL   | 95   | High    │ │
│  │ 2025-05 | KAB SLEMAN   | 78   | Medium  │ │
│  │ 2025-05 | KOTA YOGYA   | 45   | Medium  │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  [Export Predictions] [View in Dashboard]     │
└────────────────────────────────────────────────┘
```

---

## 🔧 Shall I Implement This?

I can create the complete system with:

### ✅ What I'll Build:

1. **Enhanced Data Management**
   - Separate CSV file handling
   - Upload/Download functionality
   - Web-based CRUD interface
   - Data validation
   - Automatic merging for training

2. **Complete Training Pipeline**
   - Data preparation
   - Model training interface
   - Progress tracking
   - Model versioning
   - Activation system

3. **Prediction System**
   - Generate predictions interface
   - Batch predictions
   - Results storage
   - Dashboard integration

4. **Templates & Documentation**
   - CSV templates
   - Usage guide
   - Best practices

Would you like me to implement this complete system now? 

I'll create:
- ✅ New data management routes & templates
- ✅ Enhanced training interface
- ✅ Prediction generation system
- ✅ CSV templates
- ✅ Complete documentation

Just confirm and I'll build the entire pipeline! 🚀
