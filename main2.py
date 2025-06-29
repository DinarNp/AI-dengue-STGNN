import numpy as np
import json
import time
from config.config import Config
from experiments.dengue_pipeline import DenguePredictionSystem
from models.predictor import DenguePredictor

def main():
    """Auto-run all optimization experiments and provide comprehensive results"""
    
    print("🚀 DENGUE STGNN COMPREHENSIVE OPTIMIZATION")
    print("="*60)
    print("⏱️ Starting auto-run of all experiments...")
    print("📊 This will run 4 experiments total (baseline + 3 optimizations)")
    print("⌛ Estimated time: 12-20 minutes")
    print("="*60)
    
    # Experiment configurations
    experiments = [
        {
            'name': 'BASELINE',
            'description': 'Default configuration',
            'config': {}
        },
        {
            'name': 'AGGRESSIVE_LEARNING',
            'description': 'Higher LR, Lower Dropout, Small Batch',
            'config': {
                'LEARNING_RATE': 0.0008,
                'DROPOUT': 0.15,
                'BATCH_SIZE': 8,
                'WEIGHT_DECAY': 0.0001,
                'EPOCHS': 300,  # Reduced for faster testing
                'EARLY_STOPPING_PATIENCE': 25
            }
        },
        {
            'name': 'DEEP_LEARNING_MODE',
            'description': 'Very Low Dropout, Tiny Batch, Moderate LR',
            'config': {
                'LEARNING_RATE': 0.0005,
                'DROPOUT': 0.1,
                'BATCH_SIZE': 4,
                'WEIGHT_DECAY': 0.00005,
                'EPOCHS': 400,
                'EARLY_STOPPING_PATIENCE': 30
            }
        },
        {
            'name': 'BALANCED_OPTIMIZED',
            'description': 'Conservative but Optimized Settings',
            'config': {
                'LEARNING_RATE': 0.0004,
                'DROPOUT': 0.25,
                'BATCH_SIZE': 12,
                'WEIGHT_DECAY': 0.0003,
                'EPOCHS': 250,
                'EARLY_STOPPING_PATIENCE': 20
            }
        }
    ]
    
    results = []
    start_time = time.time()
    
    # Run all experiments
    for i, experiment in enumerate(experiments):
        exp_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT {i+1}/4: {experiment['name']}")
        print(f"📋 Description: {experiment['description']}")
        print(f"{'='*80}")
        
        try:
            # Create and configure config
            config = Config()
            
            # Apply experiment-specific settings
            for key, value in experiment['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"   🔧 {key}: {value}")
            
            if not experiment['config']:
                print("   🔧 Using default configuration")
            
            # Run experiment
            print(f"\n🚀 Starting {experiment['name']} training...")
            system = DenguePredictionSystem(config)
            model, metrics, metadata = system.run_complete_pipeline("data/fix.csv")
            
            # Calculate experiment time
            exp_time = time.time() - exp_start_time
            
            # Store results
            result = {
                'experiment': experiment['name'],
                'description': experiment['description'],
                'config': experiment['config'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'zero_accuracy': metrics.get('zero_accuracy', 0),
                'prediction_stats': metrics.get('prediction_stats', {}),
                'runtime_minutes': round(exp_time / 60, 1)
            }
            results.append(result)
            
            # Immediate feedback
            print(f"\n✅ {experiment['name']} COMPLETED!")
            print(f"📊 Results: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")
            print(f"⏱️ Runtime: {result['runtime_minutes']} minutes")
            
            if i == 0:  # Baseline
                baseline_mae = metrics['mae']
                print(f"📌 Baseline MAE: {baseline_mae:.3f} (reference for comparisons)")
            else:
                improvement = ((baseline_mae - metrics['mae']) / baseline_mae) * 100
                print(f"📈 vs Baseline: {improvement:+.1f}%")
                
                if metrics['mae'] < 11.0:
                    print(f"🎯 TARGET ACHIEVED! MAE < 11.0")
                elif metrics['mae'] < baseline_mae - 0.5:
                    print(f"✅ SIGNIFICANT IMPROVEMENT!")
                elif metrics['mae'] < baseline_mae:
                    print(f"📈 IMPROVEMENT!")
                else:
                    print(f"📊 Similar to baseline")
            
        except Exception as e:
            print(f"🚨 {experiment['name']} FAILED: {str(e)}")
            result = {
                'experiment': experiment['name'],
                'error': str(e),
                'mae': float('inf'),
                'runtime_minutes': round((time.time() - exp_start_time) / 60, 1)
            }
            results.append(result)
    
    # Comprehensive Analysis
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"🎯 COMPREHENSIVE OPTIMIZATION RESULTS")
    print(f"⏱️ Total Runtime: {total_time/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Filter successful results
    successful_results = [r for r in results if r['mae'] != float('inf')]
    
    if successful_results:
        # Find best result
        best_result = min(successful_results, key=lambda x: x['mae'])
        baseline_result = next(r for r in successful_results if r['experiment'] == 'BASELINE')
        
        print(f"\n📊 PERFORMANCE RANKING:")
        sorted_results = sorted(successful_results, key=lambda x: x['mae'])
        
        for rank, result in enumerate(sorted_results, 1):
            mae = result['mae']
            
            if result['experiment'] == 'BASELINE':
                comparison = "(BASELINE)"
            else:
                improvement = ((baseline_result['mae'] - mae) / baseline_result['mae']) * 100
                comparison = f"({improvement:+.1f}% vs baseline)"
            
            status = "🏆" if result == best_result else f"{rank}."
            print(f"   {status} {result['experiment']}: MAE={mae:.3f} {comparison}")
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Method: {best_result['experiment']}")
        print(f"   Description: {best_result['description']}")
        print(f"   MAE: {best_result['mae']:.3f}")
        print(f"   RMSE: {best_result['rmse']:.3f}")
        print(f"   R²: {best_result['r2']:.3f}")
        
        # Overall improvement
        total_improvement = ((baseline_result['mae'] - best_result['mae']) / baseline_result['mae']) * 100
        print(f"   Total Improvement: {total_improvement:.1f}%")
        
        # Performance Assessment
        best_mae = best_result['mae']
        if best_mae < 9.0:
            assessment = "🎉 OUTSTANDING! Exceptional performance"
            grade = "A+"
        elif best_mae < 10.0:
            assessment = "🌟 EXCELLENT! Superior performance"
            grade = "A"
        elif best_mae < 11.0:
            assessment = "✅ VERY GOOD! Target achieved"
            grade = "B+"
        elif best_mae < 12.0:
            assessment = "📈 GOOD! Solid improvement"
            grade = "B"
        elif best_mae < 13.0:
            assessment = "📊 ACCEPTABLE! Modest improvement"
            grade = "B-"
        else:
            assessment = "⚡ BASELINE! No significant improvement"
            grade = "C"
        
        print(f"   Assessment: {assessment}")
        print(f"   Grade: {grade}")
        
        # Detailed comparison table
        print(f"\n📋 DETAILED COMPARISON TABLE:")
        print(f"{'Experiment':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Runtime':<8} {'Status'}")
        print(f"{'-'*80}")
        
        for result in sorted_results:
            if result['experiment'] == 'BASELINE':
                status = "BASELINE"
            else:
                improvement = ((baseline_result['mae'] - result['mae']) / baseline_result['mae']) * 100
                if improvement > 10:
                    status = "🔥 MAJOR"
                elif improvement > 5:
                    status = "✅ GOOD"
                elif improvement > 0:
                    status = "📈 MINOR"
                else:
                    status = "📊 SIMILAR"
            
            mae = result['mae']
            rmse = result['rmse']
            r2 = result['r2']
            runtime = f"{result['runtime_minutes']}m"
            
            print(f"{result['experiment']:<20} {mae:<8.3f} {rmse:<8.3f} {r2:<8.3f} {runtime:<8} {status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if best_result['experiment'] != 'BASELINE':
            print(f"   ✅ Use {best_result['experiment']} configuration for production")
            print(f"   📊 Expected performance: MAE ≈ {best_result['mae']:.1f} cases")
            
            # Configuration details
            print(f"   🔧 Optimal settings:")
            for key, value in best_result['config'].items():
                print(f"      {key}: {value}")
        else:
            print(f"   📊 Baseline configuration is already well-optimized")
            print(f"   🔍 Consider different model architecture or features")
        
        # Save results
        output_file = 'comprehensive_optimization_results.json'
        with open(output_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = []
            for result in results:
                json_result = result.copy()
                for key, value in json_result.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_result[key] = float(value)
                json_results.append(json_result)
            
            json.dump({
                'total_runtime_minutes': round(total_time / 60, 1),
                'best_configuration': best_result['experiment'],
                'best_mae': float(best_result['mae']),
                'total_improvement_percent': float(total_improvement),
                'results': json_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to '{output_file}'")
        print(f"🎯 Optimization completed successfully!")
        
    else:
        print(f"🚨 All experiments failed!")
    
    # Final summary for easy copy-paste
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY FOR ANALYSIS:")
    print(f"{'='*80}")
    
    if successful_results:
        print(f"Baseline MAE: {baseline_result['mae']:.3f}")
        print(f"Best MAE: {best_result['mae']:.3f}")
        print(f"Best Method: {best_result['experiment']}")
        print(f"Improvement: {total_improvement:.1f}%")
        print(f"Assessment: {grade} - {assessment}")
    
    return results

if __name__ == "__main__":
    main()