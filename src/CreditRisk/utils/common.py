import os
import sys
import pandas as pd
from src.CreditRisk.logger import logging
from src.CreditRisk.exception import CustomException
from yaml import safe_load

def read_yaml(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"YAML file not found: {filepath}")

        with open(filepath, 'r') as yaml_file:
            config = safe_load(yaml_file)  # Fixed method
            logging.info(f"✅ Successfully read YAML file")
            return config

    except Exception as e:
        logging.error(f"❌ Error in reading YAML file: {filepath}")
        raise CustomException(str(e))
    
def load_data(path):
    try:
        logging.info('Loading Data.....')
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"❌ Error in loading data {path}")
        raise CustomException(str(e))

