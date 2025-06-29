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
        """Load dengue dataset"""
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
                return df
            else:
                print("All parsing methods failed, generating synthetic data...")
                return self._generate_synthetic_data()
                    
        except Exception as e:
            print(f"Error loading data: {str(e)[:100]}")
            return self._generate_synthetic_data()
    
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
        """Create date-related features"""
        df = df.copy()
        
        if 'Year' not in df.columns or 'Week' not in df.columns:
            print("Warning: Year or Week column missing, using defaults")
            if 'Year' not in df.columns:
                df['Year'] = 2021
            if 'Week' not in df.columns:
                df['Week'] = range(1, len(df) + 1)
        
        try:
            df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + 
                                       df['Week'].astype(str).str.zfill(2) + '-1', 
                                       format='%Y-W%W-%w', errors='coerce')
            
            if df['Date'].isna().all():
                df['Date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='W')
            
            df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
            df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
            df['Month'] = df['Date'].dt.month
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            
        except Exception as e:
            print(f"Error creating date features: {e}")
            df['Date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='W')
            df['Week_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / 52)
            df['Week_cos'] = np.cos(2 * np.pi * np.arange(len(df)) / 52)
            df['Month'] = df['Date'].dt.month
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, n_lags: int = 4) -> pd.DataFrame:
        """Create lag features for cases"""
        df = df.copy()
        
        if 'Puskesmas' not in df.columns:
            df['Puskesmas'] = 'PKM_DEFAULT'
        if 'Cases' not in df.columns:
            df['Cases'] = np.random.poisson(5, len(df))
        
        try:
            df = df.sort_values(['Puskesmas', 'Date'])
            
            for lag in range(1, n_lags + 1):
                df[f'Cases_lag_{lag}'] = (df.groupby('Puskesmas')['Cases']
                                         .shift(lag).fillna(0))
                df[f'Cases_binary_lag_{lag}'] = (df[f'Cases_lag_{lag}'] > 0).astype(int)
            
            df['Cases_rolling_mean_4w'] = (df.groupby('Puskesmas')['Cases']
                                          .rolling(window=4, min_periods=1)
                                          .mean().reset_index(0, drop=True))
        except Exception as e:
            print(f"Error creating lag features: {e}")
            for lag in range(1, n_lags + 1):
                df[f'Cases_lag_{lag}'] = 0
                df[f'Cases_binary_lag_{lag}'] = 0
            df['Cases_rolling_mean_4w'] = df.get('Cases', 0)
        
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
    
    def _get_adaptive_config(self, target_mean: float) -> Dict:
        """Get adaptive training configuration based on target scale"""
        if target_mean > 8:  # High-scale dataset threshold
            return {
                'LEARNING_RATE': 0.0001,
                'DROPOUT': 0.4,
                'BATCH_SIZE': 32,
                'WEIGHT_DECAY': 0.001,
                'EPOCHS': 200,
                'PATIENCE': 15,
                'scale_type': 'high'
            }
        else:  # Low-scale dataset
            return {
                'LEARNING_RATE': 0.001,
                'DROPOUT': 0.2,
                'BATCH_SIZE': 16,
                'WEIGHT_DECAY': 0.0001,
                'EPOCHS': 300,
                'PATIENCE': 25,
                'scale_type': 'low'
            }
    
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
        val_size = int(0.1 * n_samples)
        
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
        adaptive_config = self._get_adaptive_config(target_mean)
        print(f"📋 Applied adaptive config for {adaptive_config['scale_type']} scale:")
        print(f"   Learning Rate: {adaptive_config['LEARNING_RATE']}")
        print(f"   Dropout: {adaptive_config['DROPOUT']}")
        print(f"   Batch Size: {adaptive_config['BATCH_SIZE']}")
        
        # Get node information
        if 'Puskesmas' in df.columns:
            unique_locations = df[['Puskesmas', 'Latitude', 'Longitude']].drop_duplicates()
        else:
            unique_locations = pd.DataFrame({
                'Puskesmas': ['PKM_DEFAULT'],
                'Latitude': [-7.8],
                'Longitude': [110.3]
            })
        
        location_coords = unique_locations[['Latitude', 'Longitude']].values
        
        # Prepare metadata
        metadata = {
            'feature_cols': feature_cols,
            'n_nodes': len(unique_locations),
            'node_ids': unique_locations['Puskesmas'].tolist(),
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