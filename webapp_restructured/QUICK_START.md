# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### 1. Install Dependencies (2 minutes)
```bash
cd webapp_restructured
pip install -r requirements.txt
```

### 2. Copy Your Trained Model (30 seconds)
```bash
# Copy from parent directory
cp ../dengue_stgnn_model.pth models/

# Or from experiments folder
cp ../experiment_results/experiments4/dengue_stgnn_model.pth models/
```

### 3. Run the Application (30 seconds)
```bash
python run.py
```

### 4. Login (1 minute)
1. Open browser: http://localhost:5000
2. Click "Login"
3. Use default credentials:
   - **Admin:** username: `admin`, password: `admin123`
   - **District Health:** username: `bantul`, password: `health123`

### 5. Start Using! (1 minute)

**As Admin:**
- Go to Admin → Data Management
- Try the ONE-CLICK climate data fetch
- Upload NDVI data
- Generate predictions

**As District Health:**
- Go to My District
- Update dengue cases for your regency
- View predictions and recommendations

**As Public:**
- Just visit the Dashboard (no login needed)
- View all regencies' data

---

## 🎯 Key Features to Try

### For Admin Users

#### 1. Add Dengue Cases (Manual Entry)
```
Admin → Data Management → Add Manual Entry
- Select: KAB BANTUL
- Year: 2024
- Month: 12
- Cases: 150
- Click Save
```

#### 2. Fetch Climate Data (ONE-CLICK)
```
Admin → Data Management → Climate Data section
- Year: 2024
- Month: 12
- Click "Fetch for Month"
- Wait ~30 seconds (fetches for all 5 regencies!)
```

#### 3. Export Data
```
Admin → Data Management → Export section
- Start Year: 2021
- End Year: 2024
- Click "Export to CSV"
- Downloads: data_monthly_5kab_2021_2024_ndvi.csv
```

#### 4. Generate Predictions
```
Admin → Predictions
- Year: 2025
- Month: 1
- Click "Generate for All Regencies"
- View results with risk levels
```

---

### For District Health Users

#### 1. Update Your Regency's Cases
```
My District → Update Cases
- Month: December 2024
- Cases: 100
- Notes: "Confirmed by local health center"
- Click Save
```

#### 2. View Predictions
```
My District → Predictions
- See predicted cases for next month
- View risk level (Low/Medium/High/Very High)
- Read health recommendations
```

#### 3. Request New Prediction
```
My District → Predictions → Request Prediction
- Year: 2025
- Month: 2
- Click Request
- Get instant prediction!
```

---

## 📊 Sample Data Workflow

### Complete Monthly Data Update Process

**Time Required: 5-10 minutes (vs. hours before!)**

1. **Update Dengue Cases** (2 minutes)
   ```
   Admin → Data Management
   Upload CSV with format:
   Year,Region,Month,Cases
   2024,KAB BANTUL,12,150
   2024,KAB SLEMAN,12,89
   ...
   ```

2. **Fetch Climate Data** (2 minutes)
   ```
   Admin → Data Management
   Year: 2024, Month: 12
   Click "Fetch for Month"
   System auto-fetches for all regencies
   ```

3. **Upload NDVI Data** (1 minute)
   ```
   Admin → Data Management
   Upload GeoTIFF file
   Year: 2024, Month: 12
   Click Upload
   System extracts NDVI for all regencies
   ```

4. **Generate Predictions** (30 seconds)
   ```
   Admin → Predictions
   Year: 2025, Month: 1
   Click "Generate for All Regencies"
   View results!
   ```

5. **Export (Optional)** (30 seconds)
   ```
   Admin → Data Management
   Export 2021-2024
   Download CSV for model retraining
   ```

**Total Time: ~6 minutes**

**Old Process: ~2-3 hours of running scripts!**

---

## 🔑 Default Login Credentials

### Admin Account
- **Username:** admin
- **Password:** admin123
- **Access:** Full system control

### District Health Accounts

| Regency          | Username       | Password  |
|------------------|----------------|-----------|
| KAB BANTUL       | bantul         | health123 |
| KAB GUNUNG KIDUL | gunung_kidul   | health123 |
| KAB KULON PROGO  | kulon_progo    | health123 |
| KAB SLEMAN       | sleman         | health123 |
| KOTA YOGYAKARTA  | yogyakarta     | health123 |

⚠️ **Change these passwords immediately in production!**

---

## 🎨 Interface Overview

### Navigation Structure

```
┌─────────────────────────────────────────┐
│  [Logo] Dengue Prediction System        │
│  Dashboard | Statistics | About | Login │
└─────────────────────────────────────────┘

When logged in as ADMIN:
┌─────────────────────────────────────────┐
│  Dashboard | Statistics | About | Admin │
└─────────────────────────────────────────┘
       ↓
Admin Menu:
- Dashboard (overview)
- Data Management ← KEY FEATURE
- Model Management
- Predictions
- Logs

When logged in as DISTRICT HEALTH:
┌─────────────────────────────────────────┐
│  Dashboard | Statistics | My District   │
└─────────────────────────────────────────┘
       ↓
My District Menu:
- Dashboard (your regency)
- Update Cases ← KEY FEATURE
- View Predictions
- Reports
```

---

## 📱 Key Pages to Explore

### 1. Public Dashboard (/)
**No login required**
- View dengue situation for all regencies
- See monthly trends
- Risk levels color-coded
- Provincial statistics

### 2. Admin Data Management (/admin/data-management)
**Admin only**
- Data completeness overview
- ONE-CLICK climate fetch
- NDVI upload interface
- Bulk import tools
- Export functionality

### 3. District Dashboard (/district/dashboard)
**District Health only**
- Your regency's statistics
- Recent case counts
- Latest prediction
- Monthly trend chart

### 4. Statistics Page (/statistics)
**Public access**
- Yearly comparisons
- Monthly patterns
- Regency rankings
- Key metrics

---

## 🐛 Common Issues & Solutions

### Issue: "Database not found"
**Solution:**
```bash
# Delete existing database
rm database/dengue_app.db

# Run app again - it will recreate
python run.py
```

### Issue: "Model file not found"
**Solution:**
```bash
# Ensure model is in correct location
ls models/dengue_stgnn_model.pth

# If missing, copy it
cp ../dengue_stgnn_model.pth models/
```

### Issue: "Climate API not responding"
**Solution:**
- NASA POWER API is free and doesn't need a key
- Check internet connection
- Try again (API might be temporarily down)
- System will fallback to OpenWeather if NASA fails

### Issue: "NDVI upload fails"
**Solution:**
```bash
# Install rasterio if missing
pip install rasterio

# Ensure GeoTIFF is valid format
# Verify file contains coordinate information
```

### Issue: "Prediction fails - insufficient data"
**Solution:**
- Prediction needs 4 months of historical data
- Ensure dengue cases, climate, and NDVI exist for past 4 months
- Check data completeness in Admin → Data Management

---

## 📊 Data Format Reference

### Dengue Cases CSV (for bulk import)
```csv
Year,Region,Month,Cases
2024,KAB BANTUL,1,60
2024,KAB BANTUL,2,88
2024,KAB SLEMAN,1,45
...
```

### Exported CSV (output format)
```csv
Year,Region,Month,Cases,Latitude,Longitude,NDVI,Cloud_Cover,Humidity,...
2024,KAB BANTUL,1,60,-7.902328,110.286299,0.0209,90.0,66.75,...
```

### NDVI GeoTIFF
- Format: GeoTIFF (.tif or .tiff)
- Coordinate system: WGS84 or similar
- Coverage: Should include Yogyakarta region
- Source: NASA NEO (MODIS NDVI)

---

## 🎓 Learning Path

### Day 1: Get Familiar
1. Login as admin
2. Explore the interface
3. View public dashboard
4. Check data management page

### Day 2: Add Data
1. Add manual dengue case
2. Try bulk import with sample CSV
3. Fetch climate data for one month
4. View data completeness

### Day 3: Advanced Features
1. Upload NDVI data
2. Generate predictions
3. View risk levels
4. Read recommendations

### Day 4: District Health Role
1. Login as district health user
2. Update cases for your regency
3. View predictions
4. Generate reports

### Day 5: Production Ready
1. Change default passwords
2. Upload your latest model
3. Import historical data
4. Generate predictions for next month
5. Deploy!

---

## 🚀 Next Steps

1. **Review the full README.md** for detailed documentation
2. **Test all features** with sample data
3. **Import your real data** (historical dengue cases, etc.)
4. **Upload your trained model**
5. **Change default passwords**
6. **Configure production settings**
7. **Deploy to server**

---

## 📞 Need Help?

1. Check README.md for detailed docs
2. Review code comments
3. Check application logs
4. Contact administrator

---

**Congratulations! You're ready to use the system!** 🎉

The new system makes dengue prediction accessible to everyone, not just data scientists.

---

**Tip:** Keep this guide handy for onboarding new users!
