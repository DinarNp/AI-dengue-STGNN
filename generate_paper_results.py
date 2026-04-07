from experiments.dengue_pipeline import generate_paper_results_from_checkpoint

# Generate results from your trained model
results = generate_paper_results_from_checkpoint(
    checkpoint_path='dengue_stgnn_model.pth',
    data_path='data/fix.csv'
)

print("\n✅ Results generated! Check paper_outputs/ directory")