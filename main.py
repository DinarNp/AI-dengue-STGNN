import numpy as np
from config.config import Config
from experiments.dengue_pipeline import DenguePredictionSystem
from models.predictor import DenguePredictor

def main():
    """Main function to run the dengue prediction system"""
    
    # Initialize configuration
    config = Config()
    
    # Create and run the system
    system = DenguePredictionSystem(config)
    
    # Run complete pipeline
    # You can provide actual data path here: system.run_complete_pipeline("path/to/your/data.csv")
    model, metrics, metadata = system.run_complete_pipeline("data/test2.csv")
    
    # Example of using the predictor
    print("\n" + "-" * 60)
    print("TESTING PREDICTION INTERFACE")
    print("-" * 60)
    
    # Load the saved model
    predictor = DenguePredictor('dengue_stgnn_model.pth')
    
    # Create dummy input for prediction (replace with real data)
    dummy_input = np.random.randn(config.WINDOW_SIZE, config.n_nodes if hasattr(config, 'n_nodes') else 16, 
                                 len(metadata['feature_cols']))
    
    # Make prediction
    prediction_results = predictor.predict(dummy_input)
    
    print("Prediction Results:")
    for i, node_id in enumerate(prediction_results['node_ids']):
        pred = prediction_results['predictions'][0][i]
        zero_prob = prediction_results['zero_probabilities'][0][i]
        print(f"{node_id}: Predicted cases = {pred:.2f}, Zero probability = {zero_prob:.3f}")
    
    return model, metrics, metadata

if __name__ == "__main__":
    main()