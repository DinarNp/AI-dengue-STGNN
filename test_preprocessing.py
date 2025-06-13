# test_preprocessing.py
from config.config import Config
from data.preprocessor import DengueDataPreprocessor

config = Config()
preprocessor = DengueDataPreprocessor(config)

# Load dan preprocess
df = preprocessor.load_data("data/test2.csv")
features, targets, metadata = preprocessor.preprocess_data(df)

print(f"Features shape: {features.shape}")
print(f"Targets: {targets}")
print(f"Metadata: {metadata}")