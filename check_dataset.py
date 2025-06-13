# check_new_dataset.py
from config.config import Config
from data.preprocessor import DengueDataPreprocessor

config = Config()
preprocessor = DengueDataPreprocessor(config)

# Load dataset baru
df = preprocessor.load_data("data/test2.csv")  # atau path dataset baru
features, targets, metadata = preprocessor.preprocess_data(df)

print(f"Dataset shape: {df.shape}")
print(f"Features shape: {features.shape}")
print(f"Number of nodes: {metadata['n_nodes']}")
print(f"Node IDs: {metadata['node_ids']}")
print(f"Unique puskesmas: {df['Puskesmas'].nunique() if 'Puskesmas' in df.columns else 'N/A'}")