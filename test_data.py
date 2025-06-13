from config.config import Config
from data.preprocessor import DengueDataPreprocessor

# Test loading data
config = Config()
preprocessor = DengueDataPreprocessor(config)

# Load CSV Anda
df = preprocessor.load_data("data/test2.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 2 rows:")
print(df.head(2))