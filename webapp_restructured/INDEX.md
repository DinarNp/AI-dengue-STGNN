# 📚 Documentation Index

Welcome to the restructured Dengue Prediction Web Application! This index will help you find what you need.

---

## 🚀 Getting Started

**Start here if you're new to the project:**

1. **[QUICK_START.md](QUICK_START.md)** ⭐ **START HERE**
   - 5-minute setup guide
   - Sample workflows
   - Default credentials
   - Common issues

2. **[README.md](README.md)**
   - Complete documentation (800 lines)
   - Installation instructions
   - User guides for all roles
   - API documentation

---

## 📖 Documentation Files

### For Understanding the Project

**[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**
- What was created
- Key improvements over old system
- Technical specifications
- Success metrics
- **Read this to understand the scope**

**[ARCHITECTURE.md](ARCHITECTURE.md)**
- System architecture diagrams
- Data flow charts
- Database schema
- Technology stack details
- **Read this to understand how it works**

### For Using the System

**[QUICK_START.md](QUICK_START.md)** ⭐
- Quick setup (5 minutes)
- Sample data workflows
- Interface overview
- Troubleshooting
- **Read this to start using the system**

**[README.md](README.md)**
- Detailed user guide
- Admin functions
- District health functions
- Public features
- Production deployment
- **Read this for complete usage instructions**

---

## 🎯 Find What You Need

### I want to...

#### Set up the application
→ [QUICK_START.md - Section: Get Running in 5 Minutes](QUICK_START.md#-get-running-in-5-minutes)

#### Understand what was built
→ [PROJECT_SUMMARY.md - Section: Key Achievements](PROJECT_SUMMARY.md#-key-achievements)

#### Learn how to use it as an admin
→ [README.md - Section: For Admin Users](README.md#for-admin-users)

#### Learn how to use it as district health office
→ [README.md - Section: For District Health Office Users](README.md#for-district-health-office-users)

#### View the public dashboard
→ [README.md - Section: For Public Users](README.md#for-public-users)

#### Deploy to production
→ [README.md - Section: Production Deployment](README.md#-production-deployment)

#### Understand the architecture
→ [ARCHITECTURE.md - Section: Complete System Overview](ARCHITECTURE.md#️-complete-system-overview)

#### See the data flow
→ [ARCHITECTURE.md - Section: Data Flow Diagram](ARCHITECTURE.md#-data-flow-diagram)

#### Know the API endpoints
→ [README.md - Section: API Endpoints](README.md#-api-endpoints)

#### Fix issues
→ [QUICK_START.md - Section: Common Issues & Solutions](QUICK_START.md#-common-issues--solutions)

---

## 📁 Code Organization

### Key Files

```
📄 run.py                          ← START THE APP HERE
📄 requirements.txt                ← Install dependencies
📄 config/config.py                ← Configuration settings

📄 app/__init__.py                 ← App factory
📄 app/models.py                   ← Database models (8 tables)

📁 app/routes/                     ← URL routing
   ├── auth.py                    ← Login/logout
   ├── admin.py                   ← Admin features
   ├── district_health.py         ← District health features
   └── public.py                  ← Public pages + API

📁 app/services/                   ← Business logic
   ├── data_pipeline.py           ← Data processing (700 lines)
   ├── prediction.py              ← ML predictions (300 lines)
   └── auth.py                    ← Authentication

📁 app/templates/                  ← HTML templates
📁 app/static/                     ← CSS, JS, images
📁 app/utils/                      ← Helper functions
```

---

## 🎓 Learning Path

### Day 1: Setup & Explore
1. Read [QUICK_START.md](QUICK_START.md)
2. Install and run the app
3. Login as admin (admin/admin123)
4. Explore the interface

### Day 2: Understand
1. Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Read [ARCHITECTURE.md](ARCHITECTURE.md)
3. Review the code structure
4. Check database models (app/models.py)

### Day 3: Use as Admin
1. Read [README.md - Admin section](README.md#for-admin-users)
2. Add dengue cases manually
3. Fetch climate data (one-click)
4. Upload NDVI data
5. Generate predictions

### Day 4: Use as District Health
1. Login as district user (bantul/health123)
2. Update cases for your regency
3. View predictions
4. Read recommendations

### Day 5: Deploy
1. Read [README.md - Production Deployment](README.md#-production-deployment)
2. Change default passwords
3. Configure PostgreSQL
4. Set up Gunicorn
5. Deploy!

---

## 💡 Quick Reference

### Default Credentials

**Admin:**
- Username: `admin`
- Password: `admin123`

**District Health (5 accounts):**
- Username: `bantul` | `gunung_kidul` | `kulon_progo` | `sleman` | `yogyakarta`
- Password: `health123`

⚠️ Change in production!

### Installation
```bash
cd webapp_restructured
pip install -r requirements.txt
python run.py
```

### Access
- URL: http://localhost:5000
- Public dashboard: /dashboard
- Admin panel: /admin/dashboard
- District health: /district/dashboard

### Key Features
- 🔐 Role-based access control
- 📊 Real-time predictions
- 🌡️ One-click climate data fetch
- 🛰️ Simplified NDVI processing
- 📈 Public dashboard
- 💡 Health recommendations

---

## 🔍 Search by Topic

### Authentication & Security
- User roles: [README.md - User Roles section](README.md#-default-users)
- Login process: [ARCHITECTURE.md - Authentication Flow](ARCHITECTURE.md#-authentication-flow)
- Security features: [README.md - Security Features](README.md#-security-features)

### Data Management
- Add dengue cases: [README.md - Update Dengue Cases](README.md#1-update-dengue-cases)
- Fetch climate data: [README.md - Update Climate Data](README.md#2-update-climate-data)
- Process NDVI: [README.md - Update NDVI Data](README.md#3-update-ndvi-data)
- Export data: [README.md - Export Data](README.md#4-export-data-for-model-training)

### Predictions
- How predictions work: [ARCHITECTURE.md - Prediction Flow](ARCHITECTURE.md#4-prediction-flow)
- Generate predictions: [README.md - Generate Predictions](README.md#6-generate-predictions)
- Risk levels: [README.md - Risk Visualization](README.md#risk-visualization)

### Technical Details
- System architecture: [ARCHITECTURE.md - Complete System Overview](ARCHITECTURE.md#️-complete-system-overview)
- Database schema: [ARCHITECTURE.md - Database Schema](ARCHITECTURE.md#️-database-schema)
- Data flow: [ARCHITECTURE.md - Data Flow Diagram](ARCHITECTURE.md#-data-flow-diagram)
- Tech stack: [ARCHITECTURE.md - Technology Stack](ARCHITECTURE.md#-technology-stack-details)

### Deployment
- Development setup: [QUICK_START.md](QUICK_START.md)
- Production deployment: [README.md - Production Deployment](README.md#-production-deployment)
- Configuration: [README.md - Environment Variables](README.md#environment-variables)

---

## 📊 System Comparison

### Before vs After

| Aspect | Old System | New System | Document |
|--------|-----------|------------|----------|
| Interface | Command line | Web browser | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md#1-user-experience) |
| Data pipeline | 7+ scripts | One-click | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md#2-data-management) |
| User roles | None | 3 roles | [README.md](README.md#-default-users) |
| Time to update | 70+ min | ~5 min | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md#-success-metrics) |
| Accessibility | Local only | Web accessible | [README.md](README.md#-web-interface-features) |

---

## 🛠️ Troubleshooting

### Common Issues
→ [QUICK_START.md - Common Issues & Solutions](QUICK_START.md#-common-issues--solutions)

### Detailed Troubleshooting
→ [README.md - Troubleshooting](README.md#-troubleshooting)

### Support
- Check documentation files
- Review code comments
- Check application logs
- Contact administrator

---

## 📝 File Sizes (for reference)

| File | Lines | Purpose |
|------|-------|---------|
| README.md | ~800 | Complete documentation |
| QUICK_START.md | ~500 | Quick start guide |
| PROJECT_SUMMARY.md | ~600 | Project summary |
| ARCHITECTURE.md | ~800 | Technical architecture |
| INDEX.md | ~300 | This file |

---

## 🎯 Recommended Reading Order

### For Users (Non-Technical)
1. **QUICK_START.md** (Must read)
2. **README.md** (Sections relevant to your role)
3. **PROJECT_SUMMARY.md** (Optional, for context)

### For Developers
1. **PROJECT_SUMMARY.md** (Understand what was built)
2. **ARCHITECTURE.md** (Understand how it works)
3. **README.md** (Complete reference)
4. **Code files** (app/models.py, app/services/*.py)

### For Administrators
1. **QUICK_START.md** (Get it running)
2. **README.md** (Admin section + Production deployment)
3. **ARCHITECTURE.md** (Deployment architecture)

### For Decision Makers
1. **PROJECT_SUMMARY.md** (Executive summary)
2. **README.md** (Key features section)
3. **QUICK_START.md** (See it in action)

---

## ✅ Checklist: First Time Setup

Copy and use this checklist:

- [ ] Read QUICK_START.md
- [ ] Install Python 3.8+
- [ ] Clone/copy the webapp_restructured folder
- [ ] Run: `pip install -r requirements.txt`
- [ ] Copy your trained model to models/
- [ ] Run: `python run.py`
- [ ] Access: http://localhost:5000
- [ ] Login as admin (admin/admin123)
- [ ] Explore the interface
- [ ] Try adding a dengue case
- [ ] Try fetching climate data
- [ ] Generate a prediction
- [ ] Change default passwords
- [ ] Read README.md for your role
- [ ] Import your historical data
- [ ] Deploy to production (if ready)

---

## 📞 Need More Help?

### Quick Questions
→ Check [QUICK_START.md - FAQ section](QUICK_START.md)

### User Guide
→ See [README.md - Usage Guide](README.md#-usage-guide)

### Technical Questions
→ See [ARCHITECTURE.md](ARCHITECTURE.md)

### Issues
→ Check [QUICK_START.md - Common Issues](QUICK_START.md#-common-issues--solutions)

---

## 🎊 Summary

You have access to:
- ✅ Complete web application (production-ready)
- ✅ Comprehensive documentation (2,000+ lines)
- ✅ Quick start guide
- ✅ Architecture diagrams
- ✅ User guides for all roles
- ✅ API documentation
- ✅ Deployment instructions
- ✅ Troubleshooting guides

**Everything you need is here!**

---

**Pro Tip:** Bookmark this INDEX.md file as your starting point. It will help you navigate the documentation quickly!

---

Last Updated: January 2025
Version: 1.0.0
Status: ✅ Complete

---

Happy dengue prediction! 🦟📊🎯
