import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE



from src.CreditRisk.logger import logging
from src.CreditRisk.exception import CustomException

from src.CreditRisk.config.path_config import *
from src.CreditRisk.utils.common import read_yaml,load_data

from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from scipy.stats import skew

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

         # Ensure that the directories are created if they don't exist
        PROCESSED_DIR_pre.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR_pre.mkdir(parents=True, exist_ok=True)


    # def drop_unnecessary_columns(df):
    #     columns_to_drop = ['customer_id', 'name']
    #     df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    #     logging.info(f"Dropped unnecessary columns: {columns_to_drop}")
    #     return df

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

            return df

        except Exception as e:
            logging.error("❌ Error preprocessing data")
            raise CustomException(str(e))
    
    def handle_missing_values(self,df, cat_cols, num_cols):
        # Handling missing values for categorical features
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        # Handling missing values for numerical features
        num_imputer = SimpleImputer(strategy='mean')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

        # Save imputers
        joblib.dump(cat_imputer, os.path.join(self.processed_dir, 'cat_imputer.pkl'))
        joblib.dump(num_imputer, os.path.join(self.processed_dir, 'num_imputer.pkl'))

        return df
    
    def handle_categorical_features(self,df, cat_cols):
        encoders = {}
        mappings = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            mappings[col] = {label: code for label, code in zip(le.classes_, le.transform(le.classes_))}
            logging.info(f"Label encoded column: {col}")

            # Save encoder
            joblib.dump(le, os.path.join(self.processed_dir, f'{col}_label_encoder.pkl'))

        return df
    
    def handle_numerical_features(self,df, num_cols):
        # Handle skewness: Log transformation
        skewness = df[num_cols].apply(lambda x: x.skew())
        skewed_columns = skewness[skewness > self.config['skewness_threshold']].index
        for col in skewed_columns:
            df[col] = np.log1p(df[col])  # Apply log(1+x) transformation

        # Standardize numerical features
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # Save scaler
        joblib.dump(scaler, os.path.join(self.processed_dir, 'numerical_scaler.pkl'))

        return df

    def balance_data(self,df):
        X = df.drop(columns='credit_card_default')
        y = df['credit_card_default']

        smote = SMOTE(sampling_strategy='auto', random_state=24)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Creating balanced dataframe
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df['credit_card_default'] = y_resampled

        logging.info("Data balanced using SMOTE.")
        return balanced_df
    
    def create_preprocessing_pipeline(self,cat_cols, num_cols):
        # Step 1: Define transformer for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', LabelEncoder())
        ])

        # Step 2: Define transformer for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Step 3: Combine into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ])

        return preprocessor


    def select_top_n_features(self,df, num_top_features=10):
        X = df.drop(columns='credit_card_default')
        y = df['credit_card_default']

        # Train RandomForest to get feature importances
        model = RandomForestClassifier(random_state=24)
        model.fit(X, y)

        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        })

        top_features = feature_importances.sort_values(by='Importance', ascending=False).head(num_top_features)['Feature']

        # Filter the dataframe to keep only top features
        df_top_features = df[top_features.to_list() + ['credit_card_default']]

        logging.info(f"Selected Top {num_top_features} features.")
        return df_top_features
    
    def run_data_preprocessing(self):
        # Load data
        train_df = load_data(self.train_path)
        val_df = load_data(self.val_path)

        cat_cols = self.config['categorical_columns']
        num_cols = self.config['numerical_columns']

        # Preprocess data
        train_df = self.preprocess_data(train_df)
        val_df = self.preprocess_data(val_df)

        train_df = self.handle_missing_values(train_df, cat_cols, num_cols)
        val_df = self.handle_missing_values(val_df, cat_cols, num_cols)

        train_df = self.handle_categorical_features(train_df, cat_cols)
        val_df = self.handle_categorical_features(val_df, cat_cols)

        train_df = self.handle_numerical_features(train_df, num_cols)
        val_df = self.handle_numerical_features(val_df, num_cols)

        # Optionally balance data
        train_df = self.balance_data(train_df)

        # Apply feature selection
        train_df = self.select_top_n_features(train_df)
        val_df = self.select_top_n_features(val_df)

       

        # Save processed data
        # train_df.to_csv(os.path.join(PROCESSED_TRAIN_DATA_PATH_pre, 'train_processed.csv'), index=False)
        # val_df.to_csv(os.path.join(PROCESSED_VAL_DATA_PATH_pre, 'val_processed.csv'), index=False)
        train_df.to_csv(PROCESSED_TRAIN_DATA_PATH_pre, index=False)
        val_df.to_csv(PROCESSED_VAL_DATA_PATH_pre, index=False)
        # test_df.to_csv(PROCESSED_TEST_DATA_PATH_pre, index=False)

# Testing
if __name__ == '__main__':
    logging.info('STEP 2 - DATA PREPROCESSING Initiated...')
    config = read_yaml(CONFIG_PATH)
    preprocessor = DataPreprocessing(TRAIN_FILE_PATH, VAL_FILE_PATH, PROCESSED_DIR_pre, config)
    preprocessor.run_data_preprocessing()






