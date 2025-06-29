import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

class DengueVisualizer:
    """Enhanced visualization utilities for dengue prediction results with flexible dataset support"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'warning': '#C73E1D',
            'success': '#2E8B57',
            'info': '#4A90E2'
        }
        
    def _detect_data_source_and_prepare_metadata(self, node_names, location_coords, data_path=None):
        """Intelligently detect data source and prepare enhanced metadata"""
        
        metadata = {
            'node_ids': node_names if node_names else [f'Location_{i+1}' for i in range(len(location_coords))],
            'location_coords': location_coords,
            'data_type': 'Unknown',
            'location_source': 'Generated',
            'display_names': [],
            'location_level': 'Unknown'
        }
        
        # Try to get real location data if data_path provided
        if data_path:
            try:
                df = pd.read_csv(data_path)
                
                # Detect location column hierarchy
                location_columns = ['Region', 'Puskesmas', 'Kecamatan', 'District', 'Location', 'City', 'Province']
                found_column = None
                
                for col in location_columns:
                    if col in df.columns:
                        found_column = col
                        break
                
                if found_column and 'Latitude' in df.columns and 'Longitude' in df.columns:
                    # Get unique locations with coordinates
                    location_info = df[[found_column, 'Latitude', 'Longitude']].drop_duplicates()
                    real_locations = location_info[found_column].tolist()
                    real_coords = location_info[['Latitude', 'Longitude']].values
                    
                    # Update metadata
                    metadata['node_ids'] = real_locations
                    metadata['location_coords'] = real_coords
                    metadata['data_type'] = 'Real Data'
                    metadata['location_source'] = found_column
                    
                    # Determine location level and create display names
                    metadata['location_level'] = self._determine_location_level(found_column, real_locations)
                    metadata['display_names'] = self._create_display_names(real_locations, metadata['location_level'])
                    
                    print(f"   📍 Auto-detected location data from '{found_column}' column")
                    print(f"   📊 Location level: {metadata['location_level']}")
                    print(f"   🏷️ Found {len(real_locations)} unique locations")
                    
            except Exception as e:
                print(f"   ⚠️ Could not auto-detect location data: {e}")
        
        # If no real data found, work with provided node_names
        if metadata['data_type'] == 'Unknown' and node_names:
            metadata['location_level'] = self._determine_location_level('Unknown', node_names)
            metadata['display_names'] = self._create_display_names(node_names, metadata['location_level'])
            metadata['data_type'] = 'Provided Names'
        
        return metadata
    
    def _determine_location_level(self, column_name, location_names):
        """Determine the administrative level of locations"""
        
        sample_names = [str(name).upper() for name in location_names[:3]]
        
        # Check for common patterns
        if any('PKM' in name or 'PUSKESMAS' in name for name in sample_names):
            return 'Health Facility'
        elif any('KAB' in name or 'KABUPATEN' in name for name in sample_names):
            return 'Regency'
        elif any('KOTA' in name or 'CITY' in name for name in sample_names):
            return 'City'
        elif any('KEC' in name or 'KECAMATAN' in name for name in sample_names):
            return 'District'
        elif column_name.upper() in ['REGION', 'AREA']:
            return 'Region'
        else:
            return 'Administrative Unit'
    
    def _create_display_names(self, location_names, location_level):
        """Create clean display names for visualizations"""
        
        display_names = []
        for name in location_names:
            clean_name = str(name)
            
            # Remove common prefixes/suffixes based on location level
            if location_level == 'Health Facility':
                clean_name = clean_name.replace('PKM. ', '').replace('PKM ', '').replace('PUSKESMAS ', '')
            elif location_level in ['Regency', 'City']:
                clean_name = clean_name.replace('KAB ', '').replace('KABUPATEN ', '')
                clean_name = clean_name.replace('KOTA ', '').replace('CITY ', '')
            elif location_level == 'District':
                clean_name = clean_name.replace('KEC. ', '').replace('KECAMATAN ', '')
            
            # Capitalize properly
            clean_name = clean_name.title()
            
            # Truncate if too long
            if len(clean_name) > 20:
                clean_name = clean_name[:17] + '...'
                
            display_names.append(clean_name)
        
        return display_names
    
    def plot_training_history(self, history: Dict):
        """Enhanced training history plot"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training History Dashboard', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Training Loss', 
                       color=self.colors['primary'], linewidth=2.5)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', 
                       color=self.colors['secondary'], linewidth=2.5)
        axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add best epoch marker
        best_epoch = np.argmin(history['val_loss'])
        axes[0, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, 
                          label=f'Best Epoch: {best_epoch}')
        axes[0, 0].legend()
        
        # MAE progression
        axes[0, 1].plot(history['val_mae'], label='Validation MAE', 
                       color=self.colors['accent'], linewidth=2.5)
        axes[0, 1].set_title('Validation MAE Progress', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add final MAE annotation
        final_mae = history['val_mae'][-1]
        axes[0, 1].annotate(f'Final MAE: {final_mae:.2f}', 
                           xy=(len(history['val_mae'])-1, final_mae),
                           xytext=(0.7, 0.9), textcoords='axes fraction',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='black'))
        
        # RMSE progression  
        axes[1, 0].plot(history['val_rmse'], label='Validation RMSE', 
                       color=self.colors['warning'], linewidth=2.5)
        axes[1, 0].set_title('Validation RMSE Progress', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # R² progression with zero line
        axes[1, 1].plot(history['val_r2'], label='Validation R²', 
                       color=self.colors['info'], linewidth=2.5)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='R² = 0')
        axes[1, 1].set_title('Validation R² Progress', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   📈 Enhanced training history saved as 'training_history_enhanced.png'")
    
    def plot_predictions_vs_actual(self, predictions: np.ndarray, actuals: np.ndarray,
                                  node_names: List[str] = None, data_path: str = None):
        """Enhanced predictions vs actual plot with auto-detection"""
        
        # Prepare metadata
        location_coords = np.random.uniform(-8, -7, (len(node_names) if node_names else 5, 2))
        metadata = self._detect_data_source_and_prepare_metadata(node_names, location_coords, data_path)
        
        # Call the enhanced version
        self.plot_predictions_vs_actual_enhanced(predictions, actuals, metadata)
    
    def plot_predictions_vs_actual_enhanced(self, predictions, actuals, metadata):
        """Enhanced prediction vs actual analysis with flexible dataset support"""
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Dengue Prediction Analysis - {metadata["data_type"]}', fontsize=16, fontweight='bold')
        
        # Main scatter plot
        ax1 = plt.subplot(2, 3, (1, 2))
        scatter = ax1.scatter(actuals, predictions, alpha=0.7, s=60, 
                            c=predictions, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        max_val = max(max(actuals), max(predictions))
        min_val = min(min(actuals), min(predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2.5, 
                label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Cases', fontweight='bold')
        ax1.set_ylabel('Predicted Cases', fontweight='bold')
        ax1.set_title(f'Prediction Accuracy - {metadata["location_level"]} Level', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add performance metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        r2 = 1 - (np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2))
        
        metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=11, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax1, label='Predicted Cases')
        
        # Residuals plot
        ax2 = plt.subplot(2, 3, 3)
        residuals = predictions - actuals
        ax2.scatter(predictions, residuals, alpha=0.6, s=50, color=self.colors['secondary'])
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Predicted Cases', fontweight='bold')
        ax2.set_ylabel('Residuals', fontweight='bold')
        ax2.set_title('Residuals Analysis', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add residuals statistics
        residual_std = np.std(residuals)
        ax2.fill_between([min(predictions), max(predictions)], 
                        [-2*residual_std, -2*residual_std], 
                        [2*residual_std, 2*residual_std], 
                        alpha=0.2, color='gray', label='±2σ')
        ax2.legend()
        
        # Location-wise performance
        ax3 = plt.subplot(2, 3, 4)
        n_locations = len(metadata['node_ids'])
        
        if n_locations > 1:
            samples_per_location = len(predictions) // n_locations
            location_maes = []
            
            for i in range(n_locations):
                start_idx = i * samples_per_location
                end_idx = (i + 1) * samples_per_location if i < n_locations - 1 else len(predictions)
                
                loc_pred = predictions[start_idx:end_idx]
                loc_actual = actuals[start_idx:end_idx]
                loc_mae = np.mean(np.abs(loc_pred - loc_actual))
                location_maes.append(loc_mae)
            
            # Create horizontal bar plot for better label readability
            y_pos = np.arange(len(metadata['display_names']))
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(location_maes)))
            
            bars = ax3.barh(y_pos, location_maes, color=colors, alpha=0.8, edgecolor='black')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(metadata['display_names'], fontsize=9)
            ax3.set_xlabel('MAE', fontweight='bold')
            ax3.set_title(f'Performance by {metadata["location_level"]}', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for bar, mae in zip(bars, location_maes):
                ax3.text(bar.get_width() + max(location_maes)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{mae:.1f}', ha='left', va='center', fontsize=8, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, f'Single {metadata["location_level"]}\nAnalysis\n{metadata["display_names"][0]}', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            ax3.set_title('Location Analysis', fontweight='bold')
        ax3.axis('off') if n_locations == 1 else None
        
        # Distribution comparison
        ax4 = plt.subplot(2, 3, 5)
        ax4.hist(actuals, bins=25, alpha=0.7, label='Actual Cases', 
                color=self.colors['primary'], density=True)
        ax4.hist(predictions, bins=25, alpha=0.7, label='Predicted Cases', 
                color=self.colors['accent'], density=True)
        ax4.set_xlabel('Cases', fontweight='bold')
        ax4.set_ylabel('Density', fontweight='bold')
        ax4.set_title('Distribution Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Error analysis
        ax5 = plt.subplot(2, 3, 6)
        error_percentages = np.abs(residuals) / (actuals + 1e-8) * 100  # Avoid division by zero
        ax5.hist(error_percentages, bins=25, alpha=0.7, color=self.colors['warning'], 
                edgecolor='black')
        ax5.set_xlabel('Absolute Percentage Error (%)', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Error Distribution', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add median error line
        median_error = np.median(error_percentages)
        ax5.axvline(median_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {median_error:.1f}%')
        ax5.legend()
        
        # Add data source info
        if metadata.get('location_source', 'Generated') != 'Generated':
            fig.text(0.02, 0.02, f"Data source: {metadata['location_source']} column | {metadata['location_level']} level", 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig('predictions_analysis_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   📊 Enhanced prediction analysis saved as 'predictions_analysis_enhanced.png'")
    
    def plot_spatial_heatmap(self, predictions: np.ndarray, location_coords: np.ndarray,
                           node_names: List[str] = None, data_path: str = None):
        """Enhanced spatial heatmap with auto-detection"""
        
        # Prepare metadata
        metadata = self._detect_data_source_and_prepare_metadata(node_names, location_coords, data_path)
        
        # Call the enhanced version
        self.plot_spatial_heatmap_enhanced(predictions, metadata)
    
    def plot_spatial_heatmap_enhanced(self, predictions, metadata):
        """Enhanced spatial heatmap with flexible dataset support"""
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Spatial Analysis - {metadata["data_type"]} ({metadata["location_level"]} Level)', 
                    fontsize=16, fontweight='bold')
        
        coords = metadata['location_coords']
        location_names = metadata['node_ids']
        display_names = metadata['display_names']
        
        # Calculate average predictions per location
        n_locations = len(location_names)
        samples_per_location = len(predictions) // n_locations
        
        avg_predictions = []
        for i in range(n_locations):
            start_idx = i * samples_per_location
            end_idx = (i + 1) * samples_per_location if i < n_locations - 1 else len(predictions)
            avg_pred = np.mean(predictions[start_idx:end_idx])
            avg_predictions.append(avg_pred)
        
        if len(coords) > 1:
            # Main spatial map
            ax1 = plt.subplot(2, 3, (1, 4))
            
            # Create scatter plot with size proportional to predictions
            sizes = np.array(avg_predictions) * 20 + 100  # Scale for visibility
            scatter = ax1.scatter(coords[:, 1], coords[:, 0], 
                                c=avg_predictions, s=sizes,
                                cmap='YlOrRd', alpha=0.8, 
                                edgecolors='black', linewidth=1.5)
            
            # Add location labels with improved positioning
            for i, (coord, display_name, pred) in enumerate(zip(coords, display_names, avg_predictions)):
                # Offset text to avoid overlap
                offset_x = 0.01 if i % 2 == 0 else -0.01
                offset_y = 0.01 if i < len(coords)//2 else -0.01
                
                ax1.annotate(f'{display_name}\n{pred:.1f} cases', 
                           (coord[1], coord[0]), 
                           xytext=(offset_x, offset_y), textcoords='axes fraction',
                           fontsize=9, ha='center', va='center', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   alpha=0.8, edgecolor='gray'))
            
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
            cbar.set_label('Average Predicted Cases', fontweight='bold')
            
            ax1.set_xlabel('Longitude', fontweight='bold')
            ax1.set_ylabel('Latitude', fontweight='bold')
            ax1.set_title(f'Geographic Distribution - {metadata["location_level"]}s', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add coordinate range info
            lon_range = coords[:, 1].max() - coords[:, 1].min()
            lat_range = coords[:, 0].max() - coords[:, 0].min()
            ax1.text(0.02, 0.98, f'Coverage:\n{lat_range:.2f}° lat\n{lon_range:.2f}° lon', 
                    transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    verticalalignment='top', fontsize=9)
            
            # Performance ranking table
            ax2 = plt.subplot(2, 3, 2)
            ax2.axis('tight')
            ax2.axis('off')
            
            # Create ranking data
            ranking_data = list(zip(display_names, avg_predictions))
            ranking_data.sort(key=lambda x: x[1], reverse=True)
            
            table_data = []
            for rank, (name, pred) in enumerate(ranking_data, 1):
                risk_level = self._get_risk_level(pred)
                table_data.append([rank, name, f'{pred:.1f}', risk_level])
            
            table = ax2.table(cellText=table_data,
                            colLabels=['Rank', metadata["location_level"], 'Avg Cases', 'Risk Level'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.15, 0.45, 0.2, 0.2])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            # Color code table rows
            for i in range(len(table_data)):
                risk_level = table_data[i][3]
                color = self._get_risk_color(risk_level)
                table[(i+1, 3)].set_facecolor(color)
            
            ax2.set_title('Risk Ranking', fontweight='bold')
            
            # Risk distribution pie chart
            ax3 = plt.subplot(2, 3, 3)
            risk_levels = [self._get_risk_level(pred) for pred in avg_predictions]
            risk_counts = pd.Series(risk_levels).value_counts()
            
            colors = [self._get_risk_color(level) for level in risk_counts.index]
            wedges, texts, autotexts = ax3.pie(risk_counts.values, 
                                              labels=risk_counts.index,
                                              autopct='%1.0f%%', 
                                              startangle=90,
                                              colors=colors,
                                              explode=[0.05]*len(risk_counts))
            
            # Beautify pie chart
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax3.set_title('Risk Level Distribution', fontweight='bold')
            
            # Statistical summary
            ax4 = plt.subplot(2, 3, 5)
            ax4.axis('off')
            
            stats_text = f"""
            STATISTICAL SUMMARY
            
            📊 Total {metadata["location_level"]}s: {n_locations}
            📈 Mean Prediction: {np.mean(avg_predictions):.1f} cases
            📊 Std Deviation: {np.std(avg_predictions):.1f} cases
            🔼 Maximum: {np.max(avg_predictions):.1f} cases
            🔽 Minimum: {np.min(avg_predictions):.1f} cases
            
            🎯 Coefficient of Variation: {(np.std(avg_predictions)/np.mean(avg_predictions)*100):.1f}%
            
            Risk Levels:
            🟢 Low (<5): {sum(1 for p in avg_predictions if p < 5)} locations
            🟡 Medium (5-10): {sum(1 for p in avg_predictions if 5 <= p < 10)} locations  
            🟠 High (10-20): {sum(1 for p in avg_predictions if 10 <= p < 20)} locations
            🔴 Very High (≥20): {sum(1 for p in avg_predictions if p >= 20)} locations
            """
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            
            # Trend analysis (if temporal data available)
            ax5 = plt.subplot(2, 3, 6)
            
            # Create a simple trend visualization
            x_pos = range(len(display_names))
            bars = ax5.bar(x_pos, avg_predictions, 
                          color=[self._get_risk_color(self._get_risk_level(p)) for p in avg_predictions],
                          alpha=0.8, edgecolor='black')
            
            ax5.set_xlabel('Location Index', fontweight='bold')
            ax5.set_ylabel('Predicted Cases', fontweight='bold')
            ax5.set_title('Prediction by Location', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, pred, name in zip(bars, avg_predictions, display_names):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{pred:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Customize x-axis labels
            ax5.set_xticks(x_pos[::max(1, len(x_pos)//5)])  # Show every nth label to avoid crowding
            ax5.set_xticklabels([display_names[i] for i in x_pos[::max(1, len(x_pos)//5)]], 
                               rotation=45, ha='right')
            
        else:
            # Single location analysis
            ax1 = plt.subplot(1, 1, 1)
            ax1.text(0.5, 0.5, 
                    f'Single {metadata["location_level"]} Analysis\n\n'
                    f'{display_names[0]}\n\n'
                    f'Average Prediction: {np.mean(avg_predictions):.1f} cases\n'
                    f'Risk Level: {self._get_risk_level(avg_predictions[0])}\n\n'
                    f'Coordinates: {coords[0, 0]:.3f}, {coords[0, 1]:.3f}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=16,
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            ax1.set_title(f'{metadata["location_level"]} Analysis', fontweight='bold')
            ax1.axis('off')
        
        # Add data source and metadata info
        info_text = f"Data Type: {metadata['data_type']}"
        if metadata.get('location_source', 'Generated') != 'Generated':
            info_text += f" | Source: {metadata['location_source']} column"
        info_text += f" | Level: {metadata['location_level']}"
        
        fig.text(0.02, 0.02, info_text, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                fontsize=10)
        
        plt.tight_layout()
        plt.savefig('spatial_analysis_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   🗺️ Enhanced spatial analysis saved as 'spatial_analysis_enhanced.png'")
    
    def _get_risk_level(self, prediction):
        """Determine risk level based on prediction value"""
        if prediction < 5:
            return 'Low'
        elif prediction < 10:
            return 'Medium'
        elif prediction < 20:
            return 'High'
        else:
            return 'Very High'
    
    def _get_risk_color(self, risk_level):
        """Get color for risk level"""
        color_map = {
            'Low': '#2E8B57',      # Sea Green
            'Medium': '#FFD700',    # Gold
            'High': '#FF8C00',      # Dark Orange  
            'Very High': '#DC143C'  # Crimson
        }
        return color_map.get(risk_level, '#808080')  # Gray as default