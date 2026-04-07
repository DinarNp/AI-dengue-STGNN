import pandas as pd
import numpy as np

def perform_ndvi_imputation(input_file, output_file):
    """
    Perform forward fill (ffill) and backward fill (bfill) imputation on NDVI columns
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv(input_file)
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Identify NDVI columns
    ndvi_columns = [col for col in df.columns if 'NDVI' in col.upper()]
    print(f"NDVI columns found: {ndvi_columns}")
    
    # Check missing values before imputation
    print("\nMissing values before imputation:")
    for col in ndvi_columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        print(f"{col}: {missing_count} ({missing_percentage:.2f}%)")
    
    # Create a copy of the dataframe for imputation
    df_imputed = df.copy()
    
    # Sort by Kabupaten, EpiYear, and EpiWeek to ensure proper chronological order
    print("\nSorting data chronologically...")
    df_imputed = df_imputed.sort_values(['Kabupaten', 'EpiYear', 'EpiWeek']).reset_index(drop=True)
    
    # Perform imputation for each NDVI column
    print("\nPerforming imputation...")
    
    for col in ndvi_columns:
        print(f"\nProcessing column: {col}")
        
        # Group by Kabupaten to perform imputation within each region
        def impute_group(group):
            # First apply forward fill (ffill)
            group[col] = group[col].fillna(method='ffill')
            # Then apply backward fill (bfill) for any remaining NaN values
            group[col] = group[col].fillna(method='bfill')
            return group
        
        # Apply imputation grouped by Kabupaten
        df_imputed = df_imputed.groupby('Kabupaten').apply(impute_group).reset_index(drop=True)
        
        # Check if there are still any missing values after group-wise imputation
        remaining_missing = df_imputed[col].isnull().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} values still missing after group-wise imputation")
            # Apply global ffill and bfill as fallback
            df_imputed[col] = df_imputed[col].fillna(method='ffill')
            df_imputed[col] = df_imputed[col].fillna(method='bfill')
    
    # Check missing values after imputation
    print("\nMissing values after imputation:")
    for col in ndvi_columns:
        missing_count = df_imputed[col].isnull().sum()
        missing_percentage = (missing_count / len(df_imputed)) * 100
        print(f"{col}: {missing_count} ({missing_percentage:.2f}%)")
    
    # Display some statistics about the imputed data
    print("\nImputed data statistics:")
    for col in ndvi_columns:
        if df_imputed[col].notna().any():
            print(f"\n{col}:")
            print(f"  Mean: {df_imputed[col].mean():.6f}")
            print(f"  Std: {df_imputed[col].std():.6f}")
            print(f"  Min: {df_imputed[col].min():.6f}")
            print(f"  Max: {df_imputed[col].max():.6f}")
    
    # Save the imputed data to a new CSV file
    print(f"\nSaving imputed data to {output_file}...")
    df_imputed.to_csv(output_file, index=False)
    print("Imputation completed successfully!")
    
    return df_imputed

def compare_before_after(original_df, imputed_df, ndvi_columns):
    """
    Compare the data before and after imputation
    """
    print("\n" + "="*50)
    print("COMPARISON: BEFORE vs AFTER IMPUTATION")
    print("="*50)
    
    for col in ndvi_columns:
        print(f"\n{col}:")
        
        # Original data stats
        orig_missing = original_df[col].isnull().sum()
        orig_valid = original_df[col].notna().sum()
        
        # Imputed data stats
        imp_missing = imputed_df[col].isnull().sum()
        imp_valid = imputed_df[col].notna().sum()
        
        print(f"  Before: {orig_valid} valid, {orig_missing} missing")
        print(f"  After:  {imp_valid} valid, {imp_missing} missing")
        print(f"  Imputed: {imp_valid - orig_valid} values")

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = "ndvi_complete_calendar_kabupaten_2025_2026.csv"
    output_file = "ndvi_imputed_data_2025_2026.csv"
    
    # Read original data for comparison
    print("Loading original data for comparison...")
    original_df = pd.read_csv(input_file)
    
    # Perform imputation
    imputed_df = perform_ndvi_imputation(input_file, output_file)
    
    # Compare results
    ndvi_columns = [col for col in original_df.columns if 'NDVI' in col.upper()]
    compare_before_after(original_df, imputed_df, ndvi_columns)
    
    print(f"\nProcess completed! Check '{output_file}' for the imputed data.")