#!/usr/bin/env python3
"""
Comprehensive Dengue Cases Visualization Script
================================================

Visualizes monthly dengue case data from 3 kabupaten in Yogyakarta (2013-2025)
Generates multiple publication-quality plots (300 DPI PNG files)

Date: 2026-03-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from datetime import datetime
import seaborn as sns
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
DATA_FILE = 'data/fix/monthly_case_data_dinkes_2013_2025.csv'
OUTPUT_DIR = 'visualizations'
INDIVIDUAL_DIR = os.path.join(OUTPUT_DIR, 'individual_plots')
DPI = 300  # Publication quality
FIGSIZE_COMPREHENSIVE = (20, 6)
FIGSIZE_INDIVIDUAL = (12, 6)
FIGSIZE_MA = (15, 10)

# Color scheme for kabupaten
COLORS = {
    'KAB SLEMAN': '#1f77b4',          # Blue
    'KAB GUNUNG KIDUL': '#d62728',    # Red
    'KOTA YOGYAKARTA': '#2ca02c'      # Green
}

# Moving average colors
MA_COLORS = {
    3: '#ff7f0e',   # Orange
    6: '#9467bd',   # Purple
    12: '#2ca02c'  # Green
}

MA_STYLES = {
    3: ':',    # Dotted
    6: '--',   # Dashed
    12: '-.'    # Dashdot
}

MA_WIDTHS = {
    3: 1,
    6: 1.5,
    12: 2
}


def load_and_prepare_data(filepath):
    """
    Load and prepare the dengue case data
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pandas.DataFrame with cleaned and prepared data
    """
    print(f"📁 Loading data from {filepath}...")
    
    # Load data with UTF-8 encoding to handle BOM
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Rename 'Year ' to 'Year' if needed
    if 'Year ' in df.columns:
        df.rename(columns={'Year ': 'Year'}, inplace=True)
    
    # Create proper date column
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                                 df['Month'].astype(str).str.zfill(2) + '-01')
    
    # Sort by region and date
    df = df.sort_values(['Region', 'Date']).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} records for {df['Region'].nunique()} regions")
    print(f"   Date range: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    
    return df


def calculate_moving_average(df, region, window):
    """Calculate moving average for a specific region"""
    region_data = df[df['Region'] == region].copy()
    region_data = region_data.sort_values('Date')
    region_data[f'MA_{window}'] = region_data['Cases'].rolling(window=window, center=False).mean()
    return region_data


def identify_peak_years(df, region, method='top14'):
    """
    Identify peak outbreak years for a region
    
    Args:
        df: DataFrame
        region: Region name
        method: 'top14' or 'threshold' (1.5x mean)
        
    Returns:
        List of peak years
    """
    region_data = df[df['Region'] == region].copy()
    yearly_cases = region_data.groupby('Year')['Cases'].sum()
    
    if method == 'top14':
        peak_years = yearly_cases.nlargest(14).index.tolist()
    else:  # threshold
        mean_annual = yearly_cases.mean()
        threshold = mean_annual * 1.5
        peak_years = yearly_cases[yearly_cases > threshold].index.tolist()
    
    return peak_years


def add_wet_season_shading(ax, start_year, end_year):
    """
    Add shading for wet season months (November-April)
    
    Args:
        ax: Matplotlib axis
        start_year: First year
        end_year: Last year
    """
    for year in range(start_year, end_year + 1):
        # Nov-Dec of current year
        wet_start = pd.Timestamp(f'{year}-11-01')
        wet_end = pd.Timestamp(f'{year}-12-31')
        ax.axvspan(wet_start, wet_end, alpha=0.15, color='cyan', zorder=0)
        
        # Jan-Apr of next year
        if year < end_year:
            wet_start = pd.Timestamp(f'{year+1}-01-01')
            wet_end = pd.Timestamp(f'{year+1}-04-30')
            ax.axvspan(wet_start, wet_end, alpha=0.15, color='cyan', zorder=0)


def add_peak_year_markers(ax, df, region, peak_years):
    """
    Add vertical lines for peak outbreak years at the actual peak month
    
    Args:
        ax: Matplotlib axis
        df: DataFrame  
        region: Region name
        peak_years: List of peak years
    """
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for year in peak_years:
        # Get data for this year and region
        year_data = df[(df['Region'] == region) & (df['Year'] == year)]
        
        if len(year_data) == 0:
            continue
        
        # Find the month with maximum cases in this year
        peak_month_row = year_data.loc[year_data['Cases'].idxmax()]
        peak_month = int(peak_month_row['Month'])
        peak_cases = int(peak_month_row['Cases'])
        
        # Create date at the actual peak month (middle of month for better visual)
        date = pd.Timestamp(f'{year}-{peak_month:02d}-15')
        
        # Get total cases for that year
        total_cases = int(year_data['Cases'].sum())
        
        # Draw vertical line at the peak month
        ax.axvline(date, color='red', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
        
        # Add annotation showing year, total, and peak month info
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95
        
        month_name = month_names[peak_month - 1]
        
        ax.text(date, y_pos, f'{year}\nTotal: {total_cases}\nPeak: {month_name} ({peak_cases})', 
                ha='center', va='top', fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='red'))


def plot_comprehensive_comparison(df):
    """
    Create comprehensive comparison plot with 3 kabupaten side-by-side
    """
    print("\n📊 Creating comprehensive comparison plot...")
    
    regions = df['Region'].unique()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_COMPREHENSIVE, sharey=True)
    fig.suptitle('Monthly Dengue Cases Comparison (2013-2025)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    start_year = df['Year'].min()
    end_year = df['Year'].max()
    
    for idx, (region, ax) in enumerate(zip(regions, axes)):
        region_data = df[df['Region'] == region].sort_values('Date')
        
        # Add wet season shading
        add_wet_season_shading(ax, start_year, end_year)
        
        # Identify and mark peak years
        peak_years = identify_peak_years(df, region, method='top14')
        add_peak_year_markers(ax, df, region, peak_years)
        
        # Plot the time series
        ax.plot(region_data['Date'], region_data['Cases'], 
                color=COLORS[region], linewidth=2, marker='o', markersize=3,
                label=region, alpha=0.8)
        
        # Formatting
        ax.set_title(region, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Year', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Monthly Cases', fontsize=10)
        ax.grid(True, which='major', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistics text box
        total = region_data['Cases'].sum()
        mean = region_data['Cases'].mean()
        max_val = region_data['Cases'].max()
        max_date = region_data.loc[region_data['Cases'].idxmax(), 'Date']
        
        stats_text = f'Total: {total:,}\nMean: {mean:.1f}\nMax: {max_val} ({max_date.strftime("%b %Y")})'
        ax.text(0.05, 0.85, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend for wet season
    wet_patch = mpatches.Patch(color='cyan', alpha=0.15, label='Wet Season (Nov-Apr)')
    peak_line = plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5, 
                           linewidth=1.5, label='Peak Outbreak Year')
    fig.legend(handles=[wet_patch, peak_line], loc='lower center', 
               bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'comprehensive_comparison.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_moving_averages(df):
    """
    Create moving averages trend plot (3, 6, 12 months)
    """
    print("\n📈 Creating moving averages trends plot...")
    
    regions = df['Region'].unique()
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_MA, sharex=True)
    fig.suptitle('Moving Average Trends (3, 6, and 12 Months)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    start_year = df['Year'].min()
    end_year = df['Year'].max()
    
    for idx, (region, ax) in enumerate(zip(regions, axes)):
        # Add wet season shading
        add_wet_season_shading(ax, start_year, end_year)
        
        region_data = df[df['Region'] == region].sort_values('Date')
        
        # Plot original data as light gray background
        ax.plot(region_data['Date'], region_data['Cases'], 
                color='black', linewidth=2, alpha=0.5, label='Original Data')
        
        # Calculate and plot moving averages
        for window in [3, 6, 12]:
            ma_data = calculate_moving_average(df, region, window)
            ax.plot(ma_data['Date'], ma_data[f'MA_{window}'], 
                    color=MA_COLORS[window], linestyle=MA_STYLES[window],
                    linewidth=MA_WIDTHS[window], label=f'{window}-Month MA',
                    alpha=0.9)
        
        # Formatting
        ax.set_title(region, fontsize=12, fontweight='bold', loc='left', pad=10)
        ax.set_ylabel('Cases', fontsize=10)
        ax.grid(True, which='major', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Year', fontsize=10)
    axes[-1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'moving_averages_trends.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_individual_timeseries(df, region):
    """
    Create detailed individual time series plot for one kabupaten
    """
    print(f"\n📍 Creating individual plot for {region}...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_INDIVIDUAL)
    
    region_data = df[df['Region'] == region].sort_values('Date')
    
    start_year = df['Year'].min()
    end_year = df['Year'].max()
    
    # Add wet season shading
    add_wet_season_shading(ax, start_year, end_year)
    
    # Identify and mark peak years
    peak_years = identify_peak_years(df, region, method='top14')
    add_peak_year_markers(ax, df, region, peak_years)
    
    # Plot the time series with markers
    ax.plot(region_data['Date'], region_data['Cases'], 
            color=COLORS[region], linewidth=2.5, marker='o', markersize=4,
            label='Monthly Cases', alpha=0.8, markerfacecolor='white',
            markeredgewidth=1.5, markeredgecolor=COLORS[region])
    
    # Add title and labels
    ax.set_title(f'Monthly Dengue Cases - {region} (2013-2025)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Monthly Cases', fontsize=12)
    ax.grid(True, which='major', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Calculate and display statistics
    total = region_data['Cases'].sum()
    mean = region_data['Cases'].mean()
    median = region_data['Cases'].median()
    std = region_data['Cases'].std()
    max_val = region_data['Cases'].max()
    max_date = region_data.loc[region_data['Cases'].idxmax(), 'Date']
    min_val = region_data['Cases'].min()
    
    stats_text = f'''Statistics (2013-2025):
Total Cases: {total:,}
Mean: {mean:.1f} ± {std:.1f}
Median: {median:.0f}
Max: {max_val} ({max_date.strftime("%b %Y")})
Min: {min_val}'''
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Legend
    wet_patch = mpatches.Patch(color='cyan', alpha=0.15, label='Wet Season (Nov-Apr)')
    peak_line = plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5, 
                           linewidth=1.5, label='Peak Year')
    ax.legend(handles=[wet_patch, peak_line], loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    filename = region.replace(' ', '_') + '_timeseries.png'
    output_path = os.path.join(INDIVIDUAL_DIR, filename)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_seasonal_pattern(df):
    """
    Create monthly seasonal pattern plot (average cases per month)
    """
    print("\n🌡️  Creating monthly seasonal pattern plot...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_INDIVIDUAL)
    
    regions = df['Region'].unique()
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    x = np.arange(len(months))
    width = 0.25
    
    for idx, region in enumerate(regions):
        region_data = df[df['Region'] == region]
        
        # Calculate mean and std for each month
        monthly_stats = region_data.groupby('Month')['Cases'].agg(['mean', 'std'])
        means = monthly_stats['mean'].values
        stds = monthly_stats['std'].values
        
        # Plot bars with error bars
        ax.bar(x + idx * width, means, width, 
               label=region, color=COLORS[region], alpha=0.8,
               yerr=stds, capsize=3, error_kw={'linewidth': 1, 'alpha': 0.7})
    
    # Highlight wet season months (Nov-Apr)
    wet_months = [0, 1, 2, 3, 10, 11]  # Jan, Feb, Mar, Apr, Nov, Dec (0-indexed)
    for month in wet_months:
        ax.axvspan(month - 0.4, month + 0.4, alpha=0.1, color='cyan', zorder=0)
    
    # Formatting
    ax.set_title('Average Monthly Dengue Cases by Season (2013-2025)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Cases (± Std Dev)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.grid(True, which='major', alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10)
    
    # Add wet season annotation
    ax.text(0.5, 0.98, '(Shaded areas indicate wet season months)', 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, style='italic', color='navy')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(INDIVIDUAL_DIR, 'monthly_seasonal_pattern.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def create_statistics_summary(df):
    """
    Create a visual statistics summary table
    """
    print("\n📋 Creating statistics summary table...")
    
    regions = df['Region'].unique()
    
    # Calculate statistics for each region
    stats_data = []
    for region in regions:
        region_data = df[df['Region'] == region]
        
        total = region_data['Cases'].sum()
        mean = region_data['Cases'].mean()
        median = region_data['Cases'].median()
        std = region_data['Cases'].std()
        max_val = region_data['Cases'].max()
        min_val = region_data['Cases'].min()
        max_date = region_data.loc[region_data['Cases'].idxmax(), 'Date']
        peak_years = identify_peak_years(df, region, method='top3')
        
        # Yearly trend (2013 vs 2024)
        cases_2013 = region_data[region_data['Year'] == 2013]['Cases'].sum()
        cases_2024 = region_data[region_data['Year'] == 2024]['Cases'].sum()
        if cases_2013 > 0:
            growth_rate = ((cases_2024 - cases_2013) / cases_2013) * 100
        else:
            growth_rate = 0
        
        stats_data.append({
            'Region': region,
            'Total Cases': f'{total:,}',
            'Mean ± Std': f'{mean:.1f} ± {std:.1f}',
            'Median': f'{median:.0f}',
            'Min': f'{min_val}',
            'Max': f'{max_val}',
            'Peak Date': max_date.strftime('%b %Y'),
            'Peak Years': ', '.join(map(str, peak_years)),
            '2013-2024 Change': f'{growth_rate:+.1f}%'
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    stats_df = pd.DataFrame(stats_data)
    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.1, 0.12, 0.08, 0.06, 0.06, 0.1, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_df) + 1):
        for j in range(len(stats_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            table[(i, j)].set_edgecolor('gray')
    
    plt.title('Dengue Cases Statistics Summary (2013-2025)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save
    output_path = os.path.join(INDIVIDUAL_DIR, 'statistics_summary.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()
    
    # Also print to console
    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))
    print("="*80)


def main():
    """Main execution function"""
    print("="*80)
    print("DENGUE CASES VISUALIZATION GENERATOR")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Resolution: {DPI} DPI")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(DATA_FILE)
    
    # Generate all plots
    print("\n🎨 Generating visualizations...")
    
    plot_comprehensive_comparison(df)
    plot_moving_averages(df)
    plot_seasonal_pattern(df)
    
    # Individual plots for each kabupaten
    for region in df['Region'].unique():
        plot_individual_timeseries(df, region)
    
    # Statistics summary
    create_statistics_summary(df)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*80)
    print(f"\n📂 Output files saved in: {OUTPUT_DIR}/")
    print(f"   - comprehensive_comparison.png")
    print(f"   - moving_averages_trends.png")
    print(f"   - individual_plots/KAB_SLEMAN_timeseries.png")
    print(f"   - individual_plots/KAB_GUNUNG_KIDUL_timeseries.png")
    print(f"   - individual_plots/KOTA_YOGYAKARTA_timeseries.png")
    print(f"   - individual_plots/monthly_seasonal_pattern.png")
    print(f"   - individual_plots/statistics_summary.png")
    print("\n🎉 You can now review the visualizations!")
    print("="*80)


if __name__ == "__main__":
    main()
