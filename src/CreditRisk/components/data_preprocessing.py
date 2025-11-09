import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.CreditRisk.logger import logging
from src.CreditRisk.exception import CustomException

from src.CreditRisk.config.path_config import *
from src.CreditRisk.utils.common import read_yaml,load_data

from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from scipy.stats import skew

# DATA PREPROCESSING PIPPELINE
class DataPreprocessing:
    def __init__(self, train_path, val_path, processed_dir, config):
        """Data Preprocessing - configurations"""
        self.config = config['data_preprocessing']
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = PROCESSED_TEST_DATA_PATH
        self.processed_dir = processed_dir
        
        # Create PROCESSED DIR 
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        self.transformations_dir = os.path.join(self.processed_dir, 'transformations')
        os.makedirs(self.transformations_dir, exist_ok=True)



    def preprocess_data(self, df):
        """Data Preprocessing"""
        try:
            # Drop unnecessary columns only if they exist
            columns_to_drop = ['customer_id', 'name']
            df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
            logging.info('Dropped unnecessary columns: customer_id, name')

            df['credit_card_default'] = df['credit_card_default'].astype('category')

            cat_cols = self.config['categorical_columns']
            num_cols = self.config['numerical_columns']

            # # Handle missing values before any transformations
            # df = self.handle_missing_values(df, cat_cols, num_cols)
            # df= self.handle_categroical_features(df, cat_cols)
            # df= self.handle_numerical_features(df, num_cols)
            # df = self.balance_data(df)

            # df = self.select_top_n_features(df)

            return df

        except Exception as e:
            logging.error("❌ Error preprocessing data")
            raise CustomException(str(e))
        
    def handle_categroical_features(self, df, cat_cols):
        try:
            logging.info('Encoder Handling...')
            mappings = {}
            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le 
                mappings[col] = {label:code for label,code in zip(le.classes_, le.transform(le.classes_))}
                logging.info(f"Label encoded column: {col}")


            logging.info('Label Mappings:')
            for col, mapping in mappings.items():
                logging.info(f"{col} : {mapping}")


            # Save the encoders with unique names
            for col, le in encoders.items():
                joblib.dump(le, os.path.join(self.transformations_dir, f'{col}_label_encoder.pkl'))

            return df

        except Exception as e:
            logging.error("❌ Error handling categorical data")
            raise CustomException(str(e))
    
    def handle_numerical_features(self,df,num_cols):
        try:
            logging.info('Skweness Handling...')
            skew_threshold = self.config['skewness_threshold']
            skewness = df[num_cols].apply(lambda x: x.skew())

            # Store the columns that were transformed due to skewness
            skewed_columns = []

            for col in skewness[skewness > skew_threshold].index:
                df[col] = np.log1p(df[col])
                skewed_columns.append(col)

            # Log columns transformed due to skewness
            logging.info(f"Applied log transformation to skewed columns: {', '.join(skewed_columns)}")

            # Standardizing 
            logging.info('Standardization Handling...')
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            logging.info("Standardized numerical columns")

            # Save the scaler and skewed columns to disk
            joblib.dump(scaler, os.path.join(self.transformations_dir, 'numerical_scaler.pkl'))
            joblib.dump(skewed_columns, os.path.join(self.transformations_dir, 'skewed_columns.pkl'))

            return df
        except Exception as e:
            logging.error("❌ Error handling numerical data")
            raise CustomException(str(e))



    def handle_missing_values(self, df, cat_cols, num_cols):
        """Handle missing values for categorical and numerical columns"""
        try:
            logging.info('Handling missing values...')
            
            # Handling missing values for categorical features
            cat_imputer = SimpleImputer(strategy='most_frequent')  # Most frequent category
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            logging.info(f'Handled missing values in categorical columns: {", ".join(cat_cols)}')
            
            # Handling missing values for numerical features
            num_imputer = SimpleImputer(strategy='mean')  # Mean of the column for numerical columns
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
            logging.info(f'Handled missing values in numerical columns: {", ".join(num_cols)}')

            joblib.dump(cat_imputer, os.path.join(self.transformations_dir, 'cat_imputer.pkl'))
            joblib.dump(num_imputer, os.path.join(self.transformations_dir, 'num_imputer.pkl'))

            # Log missing values after imputation
            cat_missing_info_after = df[cat_cols].isnull().sum()
            num_missing_info_after = df[num_cols].isnull().sum()
            logging.info(f"Missing values in categorical columns after imputation: {cat_missing_info_after[cat_missing_info_after > 0]}")
            logging.info(f"Missing values in numerical columns after imputation: {num_missing_info_after[num_missing_info_after > 0]}")
            
            return df
            
        except Exception as e:
            logging.error("❌ Error handling missing values")
            raise CustomException(str(e))



    def balance_data(self,df):
        try:
            logging.info('Handling Imbalanced data...')
            X= df.drop(columns='credit_card_default')
            y = df['credit_card_default']

            smote = SMOTE(sampling_strategy='auto', random_state=24)
            X_resampled, y_resampled = smote.fit_resample(X,y)
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['credit_card_default'] = y_resampled
            logging.info(f"Class distribution before SMOTE: {y.value_counts()}")
            logging.info(f"Class distribution after SMOTE: {y_resampled.value_counts()}")

            logging.info('Data Balanced successfully...')
            return balanced_df
        except Exception as e:
            logging.error("❌ Error handling Imbalanced data")
            raise CustomException(str(e))
        
    def select_top_n_features(self,df):
        try:
            logging.info('Handling Feature Selection...')

            # Drop 'customer_id' again if missed earlier
            # df = df.drop(columns=['customer_id', 'name'], axis=1, errors='ignore', inplace=True)

            X=df.drop(columns='credit_card_default')
            y=df['credit_card_default']

            model = RandomForestClassifier(random_state=24)
            model.fit(X,y)

            # Save the trained model to disk
            model_filename = os.path.join(self.transformations_dir, 'random_forest_model.pkl')
            joblib.dump(model, model_filename)
            logging.info(f'Model saved to {model_filename}')

            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            })

            top_features_importances_df = feature_importances.sort_values(by='Importance', ascending=False)

            # No of features
            top_n = self.config['no_of_features']

            top_n_features = top_features_importances_df['Feature'].head(top_n)

            top_n_df = df[top_n_features.to_list() + ['credit_card_default']]

            logging.info(f"Top {top_n} features selected: {', '.join(top_n_features)}")

            logging.info('Features selected successfully...')

            return top_n_df
        except Exception as e:
            logging.error("❌ Error selecting features")
            raise CustomException(str(e))
        

    def save_data(self, df, filepath):
        try:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            logging.info('Saving data to Processed DIR')
            df.to_csv(filepath, index=False)
            logging.info(f'Data saved to Processed DIR successfully... ,{filepath}')
        except Exception as e:
            logging.error("❌ Error saving data")
            raise CustomException(str(e))
        

    def preprocess_run(self):
        try:
            logging.info('Loading data from RAW_DIR')
            train_df = load_data(self.train_path)
            val_df = load_data(self.val_path)
            # test_df = load_data(self.test_path)

            cat_cols = self.config['categorical_columns']
            num_cols = self.config['numerical_columns']

            print("Training columns:", train_df.columns)
            print("Validation columns:", val_df.columns)

            # Apply preprocessing
            logging.info('Preprocessing training data...')
            train_df = self.preprocess_data(train_df)
            train_df = self.handle_missing_values(train_df, cat_cols, num_cols)
            train_df = self.handle_categroical_features(train_df, cat_cols)
            train_df = self.handle_numerical_features(train_df, num_cols)
            train_df = self.balance_data(train_df)

            logging.info('Preprocessing validation data...')
            val_df = self.preprocess_data(val_df)
            val_df = self.handle_missing_values(val_df, cat_cols, num_cols)
            val_df = self.handle_categroical_features(val_df, cat_cols)
            val_df = self.handle_numerical_features(val_df, num_cols)
            # val_df = self.balance_data(val_df) # Uncomment if balancing is needed for validation

            # Feature selection on train_df
            train_df = self.select_top_n_features(train_df)

            # Align the validation dataset columns to match the training set
            val_df = val_df[train_df.columns]

            # Save processed data
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(val_df, PROCESSED_VAL_DATA_PATH)
            # self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logging.info('Data Pre-processing completed successfully...')
    
        except Exception as e:
            logging.error("❌ Error running data preprocessing pipeline.....")
            raise CustomException(str(e))

# Testing
# if __name__ == '__main__':
#     logging.info('STEP 2 - DATA PREPROCESSING Initiated...')
#     config = read_yaml(CONFIG_PATH)
#     preprocessor = DataPreprocessing(TRAIN_FILE_PATH, VAL_FILE_PATH, PROCESSED_DIR, config)
#     preprocessor.preprocess_run()
