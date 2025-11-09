from pathlib import Path

# =======================================
# ========= DATA INGESTION ==============
# =======================================

# Defining base directory for raw data
RAW_DIR = Path('artifacts/raw')
RAW_FILES_PATH = RAW_DIR / 'raw_train.csv'
TRAIN_FILE_PATH = RAW_DIR / 'train.csv'
VAL_FILE_PATH = RAW_DIR / 'val.csv'
TEST_FILE_PATH = RAW_DIR / 'test.csv'

# Configuration YAML path (points to src/CreditRisk/config/configuration.yaml)
CONFIG_PATH = Path(__file__).parent / "configuration.yaml"  

# =======================================
# ========= DATA PROCESSING =============
# =======================================

# Defining processed data directory
PROCESSED_DIR = Path('artifacts/processed')
PROCESSED_DATA_DIR = PROCESSED_DIR / 'p_data'
PROCESSED_TRAIN_DATA_PATH = PROCESSED_DATA_DIR / 'processed_train.csv'
PROCESSED_VAL_DATA_PATH = PROCESSED_DATA_DIR / 'processed_val.csv'
PROCESSED_TEST_DATA_PATH = PROCESSED_DATA_DIR / 'processed_test.csv'

# =======================================
# ========= MODEL TRAINING ==============
# =======================================

# Model output directory
MODEL_OUTPUT_PATH = Path('artifacts/models')
MODEL_CM_DIR = MODEL_OUTPUT_PATH / 'images'
MODEL_BASE_DIR = MODEL_OUTPUT_PATH / 'base'
MODEL_TUNED_DIR = MODEL_OUTPUT_PATH / 'tuned'
MODEL_METRICS_DIR = MODEL_OUTPUT_PATH / 'metrics'
