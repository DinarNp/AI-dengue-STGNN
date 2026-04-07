import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DengueDataPreprocessor:
    """Handles data loading, preprocessing, and feature engineering with adaptive normalization"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load dengue dataset with enhanced temporal features"""
        try:
            # Coba beberapa separator yang mungkin
            separators = [';', ',', '\t']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    print(f"Trying separator '{sep}': shape {df.shape}")
                    
                    # Check if parsing berhasil (lebih dari 1 kolom)
                    if df.shape[1] > 1:
                        print(f"Successfully parsed with separator '{sep}'")
                        break
                except Exception as e:
                    print(f"Failed with separator '{sep}': {str(e)[:100]}")
                    continue
            
            # Jika masih gagal, coba manual parsing
            if df is None or df.shape[1] == 1:
                print("Trying manual parsing...")
                df = self._manual_csv_parsing(file_path)
            
            if df is not None and df.shape[1] > 1:
                print(f"Final data shape: {df.shape}")
                print(f"Columns: {list(df.columns[:5])}...")
                print("First row sample:")
                print(df.iloc[0, :5].to_dict())
                
                # ENHANCED: Add proper date column
                df = self._add_proper_date_column(df)
                
                return df
            else:
                print("All parsing methods failed, generating synthetic data...")
                return self._generate_synthetic_data()
                    
        except Exception as e:
            print(f"Error loading data: {str(e)[:100]}")
            return self._generate_synthetic_data()
    
    def _add_proper_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
<<<<<<< Updated upstream
        """Add proper date column from Year and Week"""
        print("🕐 Adding proper date column from Year and Week...")
        
        try:
            # Convert Year and Week to proper date
            # Week 1 of each year starts on January 1st
            df['date'] = df.apply(lambda row: 
                pd.Timestamp(year=int(row['Year']), month=1, day=1) + 
                pd.Timedelta(weeks=int(row['Week'])-1), axis=1)
            
            # Add more temporal features
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Cyclical encoding for temporal features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
=======
        """Add proper date column from Year and Week/Month"""
        
        try:
            # Check if we have Week or Month column to determine data type
            if 'Month' in df.columns:
                print("🕐 Adding proper date column from Year and Month (MONTHLY DATA)...")
                # Convert Year and Month to proper date (first day of month)
                df['date'] = pd.to_datetime(
                    df['Year'].astype(str) + '-' + 
                    df['Month'].astype(str).str.zfill(2) + '-01'
                )
                
                # Add temporal features for monthly data
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
                
                # Cyclical encoding for temporal features (monthly focus)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
                df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
                
            elif 'Week' in df.columns:
                print("🕐 Adding proper date column from Year and Week (WEEKLY DATA)...")
                # Convert Year and Week to proper date
                df['date'] = df.apply(lambda row: 
                    pd.Timestamp(year=int(row['Year']), month=1, day=1) + 
                    pd.Timedelta(weeks=int(row['Week'])-1), axis=1)
                
                # Add more temporal features for weekly data
                df['day_of_year'] = df['date'].dt.dayofyear
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
                df['week_of_year'] = df['date'].dt.isocalendar().week
                
                # Cyclical encoding for temporal features
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
                df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
                df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
                df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
            else:
                raise ValueError("Data must have either 'Week' or 'Month' column")
>>>>>>> Stashed changes
            
            # Seasonal features for Indonesia
            df['season'] = df['month'].map({
                6: 0, 7: 0, 8: 0, 9: 0,      # June-Sept: Dry season
                10: 1, 11: 1,                 # Oct-Nov: Transition
                12: 2, 1: 2, 2: 2, 3: 2,     # Dec-March: Wet/Rainy season
                4: 3, 5: 3,                   # April-May: Transition
            })
            
            # One-hot encoding for seasons
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            
            print(f"✅ Added temporal features: date, month, season, cyclical encoding")
            print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"🌡️ Seasons: {df['season'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error adding date column: {e}")
<<<<<<< Updated upstream
            # Fallback: create simple date
            df['date'] = pd.to_datetime(df['Year'].astype(str) + '-01-01') + pd.to_timedelta((df['Week']-1)*7, unit='D')
=======
            # Fallback: create simple date based on what's available
            if 'Month' in df.columns:
                # For monthly data
                df['date'] = pd.to_datetime(
                    df['Year'].astype(str) + '-' + 
                    df['Month'].astype(str).str.zfill(2) + '-01'
                )
                df['month'] = df['Month']
                df['quarter'] = ((df['Month'] - 1) // 3) + 1
            elif 'Week' in df.columns:
                # For weekly data
                df['date'] = pd.to_datetime(df['Year'].astype(str) + '-01-01') + pd.to_timedelta((df['Week']-1)*7, unit='D')
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
            else:
                # Last resort: just use year
                df['date'] = pd.to_datetime(df['Year'].astype(str) + '-01-01')
                df['month'] = 1
                df['quarter'] = 1
            
            # Add basic seasonal features
            df['season'] = df['month'].map({
                6: 0, 7: 0, 8: 0, 9: 0,
                10: 1, 11: 1,
                12: 2, 1: 2, 2: 2, 3: 2,
                4: 3, 5: 3,
            })
            
>>>>>>> Stashed changes
            return df
    
    def _manual_csv_parsing(self, file_path: str) -> pd.DataFrame:
        """Manual CSV parsing when pandas fails"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Get header
            header_line = lines[0].strip()
            
            # Try different separators for header
            if ';' in header_line:
                headers = header_line.split(';')
                separator = ';'
            elif ',' in header_line and header_line.count(',') > header_line.count(';'):
                headers = header_line.split(',')
                separator = ','
            else:
                headers = header_line.split(';')
                separator = ';'
            
            print(f"Detected {len(headers)} columns with separator '{separator}'")
            print(f"Headers: {headers[:5]}...")
            
            # Parse data rows
            data_rows = []
            for line in lines[1:]:
                if line.strip():
                    row = line.strip().split(separator)
                    if len(row) == len(headers):
                        data_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Convert numeric columns
            numeric_columns = ['Year', 'Week', 'Cases', 'Latitude', 'Longitude', 'NDVI', 
                             'Cloud_Cover', 'Humidity', 'Precipitation_Total', 
                             'Temperature_Min', 'Temperature_Max', 'Temperature_Avg', 
                             'Pressure', 'Wind_Speed', 'Wind_Direction']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Manual parsing successful: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Manual parsing failed: {e}")
            return None
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic dengue data for demonstration"""
        np.random.seed(42)
        n_centers = 3
        n_weeks = 10
        n_years = 1
        
        data = []
        center_coords = np.random.uniform(-8, -7, (n_centers, 2))
        
        for center_id in range(n_centers):
            lat, lon = center_coords[center_id]
            
            for year in range(2021, 2021 + n_years):
                for week in range(1, n_weeks + 1):
                    seasonal_factor = np.sin(2 * np.pi * week / 52) * 0.5 + 0.5
                    base_cases = np.random.poisson(seasonal_factor * 10 + 2)
                    
                    ndvi = np.random.uniform(0.2, 0.8)
                    temp_avg = 25 + 5 * seasonal_factor + np.random.normal(0, 2)
                    humidity = 60 + 20 * seasonal_factor + np.random.normal(0, 5)
                    precipitation = np.random.exponential(seasonal_factor * 50 + 10)
                    
                    data.append({
                        'Year': year,
                        'Region': 'KAB BANTUL',
                        'Source_File': f'test_file_{center_id}.xlsx',
                        'Kecamatan': f'kec_{center_id}',
                        'Puskesmas': f'PKM_{center_id:02d}',
                        'Latitude': lat,
                        'Longitude': lon,
                        'Week': week,
                        'Cases': base_cases,
                        'NDVI': ndvi,
                        'Cloud_Cover': np.random.uniform(0, 100),
                        'Humidity': humidity,
                        'Precipitation_Total': precipitation,
                        'Temperature_Min': temp_avg - 3,
                        'Temperature_Max': temp_avg + 3,
                        'Temperature_Avg': temp_avg,
                        'Pressure': 1013 + np.random.normal(0, 10),
                        'Wind_Speed': np.random.exponential(5),
                        'Wind_Direction': np.random.uniform(0, 360)
                    })
        
        df = pd.DataFrame(data)
        print(f"Generated synthetic data with shape: {df.shape}")
        return df
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date-related features - ADAPTIVE for both weekly and monthly data"""
        df = df.copy()
        
        # Check if we have monthly or weekly data
        has_month = 'Month' in df.columns
        has_week = 'Week' in df.columns
        
        if has_month and not has_week:
            # MONTHLY DATA
            print("📅 Creating date features for MONTHLY data...")
            try:
                # Create date from Year and Month (already done in _add_proper_date_column)
                if 'date' not in df.columns:
                    df['date'] = pd.to_datetime(
                        df['Year'].astype(str) + '-' + 
                        df['Month'].astype(str).str.zfill(2) + '-01'
                    )
                
                # Use existing date column
                df['Date'] = df['date']
                
                # Create Month cyclical features (if not already created)
                if 'Month_sin' not in df.columns:
                    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
                if 'Month_cos' not in df.columns:
                    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
                
                # Create Week features as ZERO for monthly data (to maintain compatibility)
                df['Week'] = 0
                df['Week_sin'] = 0
                df['Week_cos'] = 0
                
                print(f"   ✅ Created monthly features (date range: {df['date'].min()} to {df['date'].max()})")
                
            except Exception as e:
                print(f"⚠️ Error creating monthly date features: {e}")
                df['Date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='MS')  # Month start
                df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
                df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
                df['Week'] = 0
                df['Week_sin'] = 0
                df['Week_cos'] = 0
        
        elif has_week:
            # WEEKLY DATA
            print("📅 Creating date features for WEEKLY data...")
            if 'Year' not in df.columns:
                print("Warning: Year column missing, using defaults")
                df['Year'] = 2021
            
            try:
                df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + 
                                           df['Week'].astype(str).str.zfill(2) + '-1', 
                                           format='%Y-W%W-%w', errors='coerce')
                
                if df['Date'].isna().all():
                    df['Date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='W')
                
                df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
                df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
                
                # Add Month from Date
                if 'Month' not in df.columns:
                    df['Month'] = df['Date'].dt.month
                df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
                df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
                
                print(f"   ✅ Created weekly features (date range: {df['Date'].min()} to {df['Date'].max()})")
                
            except Exception as e:
                print(f"⚠️ Error creating weekly date features: {e}")
                df['Date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='W')
                df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
                df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
                df['Month'] = df['Date'].dt.month
                df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
                df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        else:
            # NO TEMPORAL DATA - use defaults
            print("⚠️ No Month or Week column found, using default temporal features")
            df['Date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='W')
            df['Week'] = df['Date'].dt.isocalendar().week
            df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
            df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
            df['Month'] = df['Date'].dt.month
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, n_lags: int = 4) -> pd.DataFrame:
        """Create enhanced lag features for cases and environmental variables"""
        df = df.copy()
        
        # Use Region instead of Puskesmas if available
        location_col = 'Region' if 'Region' in df.columns else 'Puskesmas'
        if location_col not in df.columns:
            df[location_col] = 'LOCATION_DEFAULT'
        
        if 'Cases' not in df.columns:
            df['Cases'] = np.random.poisson(5, len(df))
        
        try:
<<<<<<< Updated upstream
            df = df.sort_values([location_col, 'date'])
=======
            # Sort by date if available, otherwise by Year and Week/Month
            if 'date' in df.columns:
                df = df.sort_values([location_col, 'date'])
            elif 'Month' in df.columns:
                df = df.sort_values([location_col, 'Year', 'Month'])
            elif 'Week' in df.columns:
                df = df.sort_values([location_col, 'Year', 'Week'])
            else:
                df = df.sort_values([location_col, 'Year'])
>>>>>>> Stashed changes
            
            # Enhanced lag features for cases
            for lag in range(1, n_lags + 1):
                df[f'Cases_lag_{lag}'] = (df.groupby(location_col)['Cases']
                                         .shift(lag).fillna(0))
                df[f'Cases_binary_lag_{lag}'] = (df[f'Cases_lag_{lag}'] > 0).astype(int)
            
            # Rolling statistics for cases
            df['Cases_rolling_mean_4w'] = (df.groupby(location_col)['Cases']
                                          .rolling(window=4, min_periods=1)
                                          .mean().reset_index(0, drop=True))
            df['Cases_rolling_std_4w'] = (df.groupby(location_col)['Cases']
                                         .rolling(window=4, min_periods=1)
                                         .std().reset_index(0, drop=True).fillna(0))
            df['Cases_rolling_max_4w'] = (df.groupby(location_col)['Cases']
                                         .rolling(window=4, min_periods=1)
                                         .max().reset_index(0, drop=True))
            
            # Lag features for environmental variables
            env_vars = ['Temperature_Avg', 'Precipitation_Total', 'Humidity', 'NDVI']
            for var in env_vars:
                if var in df.columns:
                    for lag in range(1, 3):  # 1-2 week lags for environmental variables
                        df[f'{var}_lag_{lag}'] = (df.groupby(location_col)[var]
                                                 .shift(lag).fillna(df[var].median()))
            
            # Trend features
            df['Cases_trend_4w'] = (df.groupby(location_col)['Cases']
                                   .rolling(window=4, min_periods=2)
                                   .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                                   .reset_index(0, drop=True))
            
            # Outbreak indicators
            df['Cases_outbreak_indicator'] = (df['Cases'] > df.groupby(location_col)['Cases']
                                             .rolling(window=8, min_periods=4)
                                             .mean().reset_index(0, drop=True) * 2).astype(int)
            
            print(f"✅ Created enhanced lag features with {n_lags} lags")
            print(f"📊 Added rolling statistics and trend features")
            
        except Exception as e:
            print(f"Error creating lag features: {e}")
            # Fallback: create basic lag features
            for lag in range(1, n_lags + 1):
                df[f'Cases_lag_{lag}'] = 0
                df[f'Cases_binary_lag_{lag}'] = 0
            df['Cases_rolling_mean_4w'] = df.get('Cases', 0)
            df['Cases_rolling_std_4w'] = 0
            df['Cases_rolling_max_4w'] = df.get('Cases', 0)
            df['Cases_trend_4w'] = 0
            df['Cases_outbreak_indicator'] = 0
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        
        if 'NDVI' in df.columns:
            df['NDVI'] = df['NDVI'].fillna(df['NDVI'].median())
        else:
            df['NDVI'] = 0.5
        
        numeric_columns = ['Latitude', 'Longitude', 'Temperature_Avg', 'Temperature_Min',
                          'Temperature_Max', 'Humidity', 'Precipitation_Total', 'Cloud_Cover',
                          'Pressure', 'Wind_Speed', 'Wind_Direction']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
            else:
                if col in ['Latitude']:
                    df[col] = -7.8
                elif col in ['Longitude']:
                    df[col] = 110.3
                elif col in ['Temperature_Avg', 'Temperature_Min', 'Temperature_Max']:
                    df[col] = 26.0
                elif col in ['Humidity']:
                    df[col] = 65.0
                elif col in ['Pressure']:
                    df[col] = 1013.0
                else:
                    df[col] = 0.0
        
        return df
    
    def _get_adaptive_config(self, target_mean: float, n_locations: int = 5) -> Dict:
        """Create adaptive configuration - FIXED VERSION"""
        
        if target_mean > 8.0:  # High-scale data
            print(f"🔄 High-scale data detected - applying log1p normalization")
            
            # ✅ BETTER HIGH-SCALE CONFIG
            adaptive_config = {
                'LEARNING_RATE': 0.0005,      # ✅ Higher LR for better learning
                'DROPOUT': 0.15,              # ✅ Lower dropout (was 0.4)
                'BATCH_SIZE': 8,             # ✅ Smaller batches (was 32)
                'WEIGHT_DECAY': 1e-6,        # ✅ Less regularization
                'EPOCHS': 1000,               # ✅ More epochs (was 200)
                'PATIENCE': 100,              # ✅ More patience (was 15)
                'GRADIENT_CLIP': 1.0,
                'HIDDEN_DIM': 256,
                'NUM_LAYERS': 4,
                'NUM_HEADS': 4,
                'K_NEIGHBORS': 3,            # ✅ Reduce graph connectivity
                'scale_type': 'high'
            }
            
            print(f"📋 Applied adaptive config for high scale:")
            print(f"   Learning Rate: {adaptive_config['LEARNING_RATE']}")
            print(f"   Dropout: {adaptive_config['DROPOUT']}")
            print(f"   Batch Size: {adaptive_config['BATCH_SIZE']}")
            print(f"   Epochs: {adaptive_config['EPOCHS']}")
            print(f"   Patience: {adaptive_config['PATIENCE']}")
            
        else:  # Low-scale data
            adaptive_config = {
                'LEARNING_RATE': 0.001,
                'DROPOUT': 0.15,
                'BATCH_SIZE': 16,
                'WEIGHT_DECAY': 1e-6,
                'EPOCHS': 300,
                'PATIENCE': 30,
                'GRADIENT_CLIP': 1.0,
                'HIDDEN_DIM': 128,
                'NUM_LAYERS': 3,
                'NUM_HEADS': 4,
                'K_NEIGHBORS': 4,
                'scale_type': 'low'
            }
        
        return adaptive_config

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Complete preprocessing pipeline with adaptive target normalization - FIXED VERSION"""
        print("Starting adaptive data preprocessing...")
        
        # Create features
        df = self.create_date_features(df)
        df = self.create_lag_features(df)
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        if 'Kecamatan' in df.columns:
            try:
                df['Kecamatan_encoded'] = self.label_encoder.fit_transform(df['Kecamatan'].astype(str))
            except:
                df['Kecamatan_encoded'] = 0
        else:
            df['Kecamatan_encoded'] = 0
        
        # Get unique locations count
        if 'Region' in df.columns:
            n_locations = df['Region'].nunique()
        else:
            n_locations = 5  # Default
            
        # Define feature columns
        feature_cols = [
            'Latitude', 'Longitude', 'NDVI', 'Temperature_Avg', 'Temperature_Min',
            'Temperature_Max', 'Humidity', 'Precipitation_Total', 'Cloud_Cover',
            'Pressure', 'Wind_Speed', 'Wind_Direction', 'Week_sin', 'Week_cos',
            'Month_sin', 'Month_cos', 'Cases_lag_1', 'Cases_lag_2', 'Cases_lag_3',
            'Cases_lag_4', 'Cases_binary_lag_1', 'Cases_binary_lag_2',
            'Cases_binary_lag_3', 'Cases_binary_lag_4', 'Cases_rolling_mean_4w',
            'Kecamatan_encoded'
        ]
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Scale features
        try:
            features_scaled = self.scaler.fit_transform(df[feature_cols])
        except Exception as e:
            print(f"Error scaling features: {e}")
            features_scaled = df[feature_cols].values
        
        # Get target values
        if 'Cases' in df.columns:
            target_values = df['Cases'].values
        else:
            target_values = np.random.poisson(5, len(df))
        
        # 🎯 SINGLE TARGET ANALYSIS (FIXED - NO DUPLICATION)
        target_mean = target_values.mean()
        target_max = target_values.max()
        target_std = target_values.std()
        zero_count = (target_values == 0).sum()
        
        print(f"📊 Target analysis (full dataset):")
        print(f"   Mean: {target_mean:.2f}, Max: {target_max}, Std: {target_std:.2f}")
        print(f"   Zeros: {zero_count}/{len(target_values)} ({zero_count/len(target_values)*100:.1f}%)")
        print(f"   Percentiles: 25%={np.percentile(target_values, 25):.1f}, 50%={np.percentile(target_values, 50):.1f}, 75%={np.percentile(target_values, 75):.1f}")
        
        # 🎯 ANALYZE DATA SPLIT EFFECT
        n_samples = len(target_values)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        train_targets = target_values[:train_size]
        val_targets = target_values[train_size:train_size + val_size]
        test_targets = target_values[train_size + val_size:]
        
        print(f"📋 Data split analysis:")
        print(f"   Train: mean={train_targets.mean():.2f}, max={train_targets.max():.0f}")
        print(f"   Val: mean={val_targets.mean():.2f}, max={val_targets.max():.0f}")
        print(f"   Test: mean={test_targets.mean():.2f}, max={test_targets.max():.0f}")
        
        # 🎯 SINGLE NORMALIZATION DECISION (FIXED)
        if target_mean > 8:  # High-scale threshold
            print("🔄 High-scale data detected - applying log1p normalization")
            target_values_normalized = np.log1p(target_values)
            
            transform_info = {
                'type': 'log1p',
                'original_mean': float(target_mean),
                'original_max': float(target_max),
                'original_std': float(target_std),
                'normalized_mean': float(target_values_normalized.mean()),
                'normalized_max': float(target_values_normalized.max()),
                'normalized_std': float(target_values_normalized.std()),
                'split_stats': {
                    'train_mean': float(train_targets.mean()),
                    'val_mean': float(val_targets.mean()),
                    'test_mean': float(test_targets.mean())
                }
            }
            
            print(f"   After log1p: mean={transform_info['normalized_mean']:.2f}, max={transform_info['normalized_max']:.2f}")
            
        else:
            print("📈 Low-scale data - keeping original values")
            target_values_normalized = target_values
            transform_info = {
                'type': 'none',
                'original_mean': float(target_mean),
                'original_max': float(target_max),
                'original_std': float(target_std),
                'split_stats': {
                    'train_mean': float(train_targets.mean()),
                    'val_mean': float(val_targets.mean()),
                    'test_mean': float(test_targets.mean())
                }
            }
        
        # 🎯 GET ADAPTIVE CONFIG
        adaptive_config = self._get_adaptive_config(target_mean, n_locations)
        print(f"📋 Applied adaptive config for {adaptive_config['scale_type']} scale:")
        print(f"   Learning Rate: {adaptive_config['LEARNING_RATE']}")
        print(f"   Dropout: {adaptive_config['DROPOUT']}")
        print(f"   Batch Size: {adaptive_config['BATCH_SIZE']}")
        
        # Get node information
        if 'Region' in df.columns:
            unique_locations = df[['Region', 'Latitude', 'Longitude']].drop_duplicates()
            print(f"✅ Found {len(unique_locations)} unique regions for graph construction")
        elif 'Puskesmas' in df.columns:
            unique_locations = df[['Puskesmas', 'Latitude', 'Longitude']].drop_duplicates()
            print(f"✅ Found {len(unique_locations)} unique puskesmas for graph construction")
        else:
            unique_locations = pd.DataFrame({
                'Region': ['DEFAULT_REGION'],
                'Latitude': [-7.8],
                'Longitude': [110.3]
            })
            print("⚠️ No location column found, using default coordinates")
        
        location_coords = unique_locations[['Latitude', 'Longitude']].values
        
        # Prepare metadata
        # Get node_ids based on available column
        if 'Region' in unique_locations.columns:
            node_ids = unique_locations['Region'].tolist()
        else:
            node_ids = unique_locations['Puskesmas'].tolist()
        
        metadata = {
            'feature_cols': feature_cols,
            'n_nodes': len(unique_locations),
            'node_ids': node_ids,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'location_coords': location_coords,
            'target_transform': transform_info['type'],
            'target_stats': transform_info,
            'adaptive_config': adaptive_config
        }
        
        print(f"✅ Preprocessing complete:")
        print(f"   Features shape: {features_scaled.shape}")
        print(f"   Targets shape: {target_values_normalized.shape}")
        print(f"   Number of nodes: {metadata['n_nodes']}")
        print(f"   Target transform: {metadata['target_transform']}")

        self.debug_data_split_detailed(df, target_values)
        
        return features_scaled, target_values_normalized, metadata
    
    def analyze_dataset_characteristics(self, df: pd.DataFrame, targets: np.ndarray):
        """Comprehensive dataset analysis for paper reporting"""
        
        print("\n" + "="*80)
        print("DATASET CHARACTERISTICS ANALYSIS")
        print("="*80)
        
        # Basic statistics
        stats = {
            'total_records': len(targets),
            'mean': targets.mean(),
            'median': np.median(targets),
            'std': targets.std(),
            'min': targets.min(),
            'max': targets.max(),
            'zero_count': np.sum(targets == 0),
            'zero_percentage': np.sum(targets == 0) / len(targets) * 100
        }
        
        print(f"\n📊 Overall Target Statistics:")
        print(f"   Total records: {stats['total_records']}")
        print(f"   Mean: {stats['mean']:.2f} cases/week")
        print(f"   Median: {stats['median']:.2f}")
        print(f"   Std Dev: {stats['std']:.2f}")
        print(f"   Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"   Zero observations: {stats['zero_count']} ({stats['zero_percentage']:.1f}%)")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95]
        perc_values = np.percentile(targets, percentiles)
        print(f"\n   Percentiles:")
        for p, v in zip(percentiles, perc_values):
            print(f"      P{p}: {v:.1f}")
        
        # Seasonal analysis
        if 'Week' in df.columns:
            df_temp = df.copy()
            df_temp['Cases'] = targets
            weekly_stats = df_temp.groupby('Week')['Cases'].agg(['mean', 'std', 'count'])
            
            season_variation = weekly_stats['mean'].std() / weekly_stats['mean'].mean()
            print(f"\n📅 Seasonal Analysis:")
            print(f"   Seasonal coefficient of variation: {season_variation:.2f}")
            
            # Identify peak weeks
            high_weeks = weekly_stats[weekly_stats['mean'] > weekly_stats['mean'].quantile(0.75)].index.tolist()
            low_weeks = weekly_stats[weekly_stats['mean'] < weekly_stats['mean'].quantile(0.25)].index.tolist()
            print(f"   Peak transmission weeks: {high_weeks[:5]}...")
            print(f"   Low transmission weeks: {low_weeks[:5]}...")
        
        # Location analysis
        if 'Region' in df.columns:
            df_temp = df.copy()
            df_temp['Cases'] = targets
            location_stats = df_temp.groupby('Region')['Cases'].agg(['mean', 'std', 'min', 'max', 'count'])
            
            print(f"\n🗺️ Spatial Heterogeneity:")
            for region in location_stats.index:
                stats = location_stats.loc[region]
                print(f"   {region}:")
                print(f"      Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                print(f"      Range: [{stats['min']:.0f}, {stats['max']:.0f}], N={stats['count']}")
            
            location_cv = location_stats['mean'].std() / location_stats['mean'].mean()
            print(f"   Location coefficient of variation: {location_cv:.2f}")
        
        return stats

    def debug_data_split_detailed(self, df: pd.DataFrame, target_values: np.ndarray):
        """Comprehensive debug analysis of data distribution"""
        
        print("\n" + "="*70)
        print("🔍 COMPREHENSIVE DATA ANALYSIS")
        print("="*70)
        
        # Overall statistics
        print(f"📊 Overall dataset:")
        print(f"   Total samples: {len(target_values)}")
        print(f"   Mean: {target_values.mean():.2f}")
        print(f"   Std: {target_values.std():.2f}")
        print(f"   Min: {target_values.min():.2f}, Max: {target_values.max():.2f}")
        print(f"   Zeros: {np.sum(target_values == 0)} ({np.sum(target_values == 0)/len(target_values)*100:.1f}%)")
        
        # Percentile analysis
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        perc_values = np.percentile(target_values, percentiles)
        print(f"   Percentiles: " + ", ".join([f"P{p}={v:.1f}" for p, v in zip(percentiles, perc_values)]))
        
        # Check temporal patterns
        if 'Week' in df.columns and 'Year' in df.columns:
            print(f"\n📅 Temporal Analysis:")
            
            # Group by week and calculate mean cases
            df_temp = df.copy()
            df_temp['Cases'] = target_values
            
            week_stats = df_temp.groupby('Week')['Cases'].agg(['mean', 'std', 'count']).round(2)
            print(f"   Week-wise statistics (sample):")
            print(f"   Week 1-5: {week_stats.head()['mean'].tolist()}")
            
            # Check if there's seasonal pattern
            weekly_means = week_stats['mean'].values
            season_variation = np.std(weekly_means) / np.mean(weekly_means)
            print(f"   Seasonal variation coefficient: {season_variation:.2f}")
            
            if season_variation > 0.5:
                print(f"   🚨 HIGH seasonal variation detected!")
                
                # Find high/low seasons
                high_weeks = week_stats[week_stats['mean'] > np.percentile(weekly_means, 75)].index.tolist()
                low_weeks = week_stats[week_stats['mean'] < np.percentile(weekly_means, 25)].index.tolist()
                
                print(f"   High-case weeks: {high_weeks}")
                print(f"   Low-case weeks: {low_weeks}")
        
        # Check location patterns  
        if 'Region' in df.columns:
            print(f"\n🗺️ Location Analysis:")
            
            df_temp = df.copy()
            df_temp['Cases'] = target_values
            
            location_stats = df_temp.groupby('Region')['Cases'].agg(['mean', 'std', 'count']).round(2)
            print(f"   Location statistics:")
            for region in location_stats.index:
                stats = location_stats.loc[region]
                print(f"   {region}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, samples={stats['count']}")
            
            # Check variation between locations
            location_means = location_stats['mean'].values
            location_variation = np.std(location_means) / np.mean(location_means)
            print(f"   Location variation coefficient: {location_variation:.2f}")
            
            if location_variation > 0.3:
                print(f"   ⚠️ Significant variation between locations!")
        
        # Simulate different split strategies and compare
        print(f"\n🎲 Split Strategy Comparison:")
        
        n_samples = len(target_values)
        
        # Strategy 1: Time-based (original problematic method)
        train_size = int(0.7 * n_samples)
        val_size = int(0.1 * n_samples)
        
        time_train = target_values[:train_size]
        time_val = target_values[train_size:train_size + val_size]
        time_test = target_values[train_size + val_size:]
        
        print(f"   Time-based split:")
        print(f"      Train: mean={time_train.mean():.2f}, std={time_train.std():.2f}")
        print(f"      Val:   mean={time_val.mean():.2f}, std={time_val.std():.2f}")
        print(f"      Test:  mean={time_test.mean():.2f}, std={time_test.std():.2f}")
        print(f"      Test bias: {abs(time_test.mean() - target_values.mean())/target_values.mean()*100:.1f}%")
        
        # Strategy 2: Random split
        from sklearn.model_selection import train_test_split
        train_idx, temp_idx = train_test_split(range(n_samples), test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.67, random_state=42)
        
        random_train = target_values[train_idx]
        random_val = target_values[val_idx]
        random_test = target_values[test_idx]
        
        print(f"   Random split:")
        print(f"      Train: mean={random_train.mean():.2f}, std={random_train.std():.2f}")
        print(f"      Val:   mean={random_val.mean():.2f}, std={random_val.std():.2f}")
        print(f"      Test:  mean={random_test.mean():.2f}, std={random_test.std():.2f}")
        print(f"      Test bias: {abs(random_test.mean() - target_values.mean())/target_values.mean()*100:.1f}%")
        
        # Strategy 3: Stratified by quartiles
        try:
            target_quartiles = np.digitize(target_values, bins=np.percentile(target_values, [25, 50, 75]))
            strat_train_idx, strat_temp_idx = train_test_split(range(n_samples), test_size=0.3, 
                                                            stratify=target_quartiles, random_state=42)
            strat_val_idx, strat_test_idx = train_test_split(strat_temp_idx, test_size=0.67, 
                                                            stratify=target_quartiles[strat_temp_idx], random_state=42)
            
            strat_train = target_values[strat_train_idx]
            strat_val = target_values[strat_val_idx]
            strat_test = target_values[strat_test_idx]
            
            print(f"   Stratified split:")
            print(f"      Train: mean={strat_train.mean():.2f}, std={strat_train.std():.2f}")
            print(f"      Val:   mean={strat_val.mean():.2f}, std={strat_val.std():.2f}")
            print(f"      Test:  mean={strat_test.mean():.2f}, std={strat_test.std():.2f}")
            print(f"      Test bias: {abs(strat_test.mean() - target_values.mean())/target_values.mean()*100:.1f}%")
            
        except Exception as e:
            print(f"   Stratified split failed: {e}")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATIONS:")
        
        time_bias = abs(time_test.mean() - target_values.mean())/target_values.mean()
        random_bias = abs(random_test.mean() - target_values.mean())/target_values.mean()
        
        if time_bias > 0.3:
            print(f"   🚨 Time-based split has SEVERE bias ({time_bias:.1%})")
            if random_bias < time_bias * 0.5:
                print(f"   ✅ RECOMMENDED: Use random split (bias only {random_bias:.1%})")
            else:
                print(f"   ⚠️ All splits have bias issues - data may be inherently skewed")
        
        if 'Week' in df.columns and season_variation > 0.5:
            print(f"   📅 High seasonal variation detected - consider:")
            print(f"      1. Group by season and split each season")
            print(f"      2. Use time series cross-validation")
            print(f"      3. Add seasonal features to model")
        
        print("="*70)