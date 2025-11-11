import os
import pandas as pd
from google.cloud import storage
from src.CreditRisk.logger import logging
from src.CreditRisk.exception import CustomException
from src.CreditRisk.config.path_config import *
from src.CreditRisk.utils.common import read_yaml

from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, config):
        """Data Ingestion - configurations"""
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.filenames = self.config['bucket_file_names']
        self.train_val_ratio = self.config['val_ratio']
        self.rnd_state = self.config['random_state']

        # Create RAW DIR 
        os.makedirs(RAW_DIR, exist_ok=True)
        logging.info(f'Data Ingestion started with {self.bucket_name} & file : {self.filenames}')

    # Download Data from GCP Bucket
    def download_data(self):
        """Download files from GCP bucket"""
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            for file in self.filenames:
                blob = bucket.blob(file)
                dest_path = os.path.join(RAW_DIR, file)
                blob.download_to_filename(dest_path)
                logging.info(f"✅ Downloaded {file} to {dest_path}")
        except Exception as e:
            logging.error("❌ Error downloading data")
            raise CustomException(str(e))
        
    
    # Split Data into Train and validation SETS
    def split_train_val(self):
        """Split train data into train/validation"""
        try:
            train_file = os.path.join(RAW_DIR, 'train.csv')
            train_df = pd.read_csv(train_file)

            train_data, val_data = train_test_split(train_df, test_size=self.train_val_ratio, 
                                                    random_state=self.rnd_state)
            
            # Check the columns before saving the split
            print("Before split - Train columns:", train_data.columns)
            print("Before split - Validation columns:", val_data.columns)

            # Ensure that both train and validation have the same columns
            if not (train_data.columns == val_data.columns).all():
                print("⚠️ Column mismatch between train and validation sets!")

            # Save the split datasets to CSV
            os.makedirs(os.path.dirname(TRAIN_FILE_PATH), exist_ok=True)
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            val_data.to_csv(VAL_FILE_PATH, index=False)

            # Log success message
            logging.info(f"✅ Train/Validation split done. Train: {train_data.shape}, Val: {val_data.shape}")

            logging.info(f"✅ Train/Validation split done. Train: {train_data.shape}, Val: {val_data.shape}")
        except Exception as e:
            logging.error("❌ Error splitting train/validation data")
            raise CustomException(str(e))
        

    def run(self):
        try:
            self.download_data()
            self.split_train_val()
            logging.info('✅ Data Ingestion completed successfully...')
        except Exception as e:
            logging.error("❌ Error in Data Ingestion Step")
            raise CustomException(str(e))
        finally:
            logging.info(' Data Ingestion Ended..........')


# if __name__=='__main__':
#     # Load config
#     logging.info('Step 1: Data Ingestion')
#     config = read_yaml(CONFIG_PATH)
#     data_ingestion = DataIngestion(config)
#     data_ingestion.run()