from src.CreditRisk.logger import logging
from src.CreditRisk.exception import CustomException

from src.CreditRisk.config.path_config import *
from src.CreditRisk.utils.common import read_yaml
from src.CreditRisk.config.params import *

from src.CreditRisk.components.data_ingestion import DataIngestion
from src.CreditRisk.components.data_preprocessing import DataPreprocessing
from src.CreditRisk.components.model_training import ModelTraining

def main():
    try:
        logging.info(f'>>>>>>>>>> ML PIPELINE <<<<<<<<<<')
        # Load config
        logging.info('>>>>>>>>>> Step 1: DATA INGESTION <<<<<<<<<<')
        config = read_yaml(CONFIG_PATH)
        data_ingestion = DataIngestion(config)
        data_ingestion.run()

        logging.info('>>>>>>>>>> Step 2: DATA PREPROCESSING <<<<<<<<<<')
        config = read_yaml(CONFIG_PATH)
        preprocessor = DataPreprocessing(TRAIN_FILE_PATH, VAL_FILE_PATH, PROCESSED_DIR, config)
        preprocessor.preprocess_run()

        logging.info('>>>>>>>>>> Step 3: MODEL TRAINING <<<<<<<<<<')
        config = read_yaml(CONFIG_PATH)
        model_trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_VAL_DATA_PATH,
                                        MODEL_OUTPUT_PATH, config)
        model_trainer.train_run()

        logging.info(f'>>>>>>>>>> ML PIPELINE COMPLETED... <<<<<<<<<<')
    except Exception as e:
            logging.error("❌ FAILED - MODEL TRAINING PIPELINE")
            raise CustomException(str(e))
    
if __name__=='__main__':
     main()