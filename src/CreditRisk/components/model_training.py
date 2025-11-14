import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from src.CreditRisk.logger import logging
from src.CreditRisk.exception import CustomException

from src.CreditRisk.config.path_config import *
from src.CreditRisk.utils.common import read_yaml,load_data
from src.CreditRisk.config.params import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score,precision_score,recall_score,
                             f1_score,roc_auc_score, log_loss,
                             confusion_matrix,classification_report)
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse
from dotenv import load_dotenv
from scipy.special import expit
from scipy.stats import randint,uniform


class ModelTraining:
    def __init__(self, train_path, val_path, model_dir, config):
        self.config = config['model_training']
        self.train_path = train_path 
        self.val_path = val_path  
        self.model_dir = model_dir
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        self.model_cm_dir = os.path.join(self.model_dir,'images')
        self.top_n = self.config['top_n']
        self.model_base_dir = os.path.join(self.model_dir,'base')
        self.model_tuned_dir = os.path.join(self.model_dir,'tuned')
        self.model_metrics_dir = os.path.join(self.model_dir,'metrics')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_split_data(self):
        """Load - SPLIT data into X/y"""
        try:
            logging.info(f"Loading TRAIN data: {self.train_path}")
            # Load data first
            train_df = load_data(self.train_path)
            val_df = load_data(self.val_path)
            
            feature_columns = [
                "credit_score", "credit_limit_used", "prev_defaults", 
                "default_in_last_6months", "no_of_days_employed", "yearly_debt_payments", 
                "age", "net_yearly_income", "credit_limit", "owns_car"
            ]

            X_train = train_df[feature_columns]  # Explicitly use only these features
            y_train = train_df['credit_card_default']
            X_val = val_df[feature_columns]  # Explicitly use only these features for validation
            y_val = val_df['credit_card_default']

            logging.info("Data splitted successfully for Model Training")
            return X_train, y_train, X_val, y_val
        except Exception as e:
            logging.error("❌ Error loading-splitting data")
            raise CustomException(str(e))

        
       
        
    def train_lgbm(self,X_train,y_train):
        """Train LightGBM with RandomizedSearchCV"""
        try:
            logging.info("Intializing our model")

            lgbm_model = LGBMClassifier(random_state=self.random_search_params["random_state"])

            logging.info("Starting our Hyperparamter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )

            random_search.fit(X_train,y_train)

            logging.info("Hyperparamter tuning completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logging.info(f"Best paramters are : {best_params}")

            return best_lgbm_model, best_params
        
        except Exception as e:
            logging.error(f"Error while training model: {str(e)}")
            raise CustomException("Failed to train model" ,  e)
    

    def evaluate_model(self, model, X_val, y_val, stage='baseline'):
        try:
            # Predictions
            y_pred = model.predict(X_val)
            y_prob = None

            # Get probabilities for ROC AUC / log_loss
            if hasattr(model, 'predict_proba'):
                y_prob_full = model.predict_proba(X_val)
                y_prob = y_prob_full[:,1]
            elif hasattr(model, 'decision-function'):
                y_prob = model.decision_function(X_val)
                y_prob = expit(y_prob)

            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_prob),
                'log_loss': log_loss(y_val, y_prob),
                'class_report': classification_report(y_val, y_pred)
            }

            # ROC AUC
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_val, y_prob)
                metrics['log_loss'] = log_loss(y_val, y_prob_full if hasattr(model, "predict_proba") else y_prob)

            # Print metrics for debugging
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"F1 Score: {metrics['f1']}")
            print(f"Precision: {metrics['precision']}")
            print(f"Recall: {metrics['recall']}")
            print(f"Classification Report: {metrics['class_report']}")

            if 'roc_auc' in metrics:
                print(f"ROC AUC: {metrics['roc_auc']}")
            if 'log_loss' in metrics:
                print(f"Log Loss: {metrics['log_loss']}")

            # Save metrics to JSON file
            metrics_filename = f"{model.__class__.__name__}_metrics_{stage}.json"
            metrics_filepath = os.path.join(self.model_metrics_dir, metrics_filename)

            # Save classification report to a text file
            classification_report_str = metrics['class_report']
            report_file_path = os.path.join(self.model_metrics_dir, f'{model.__class__.__name__}_class_report.txt')
            with open(report_file_path, 'w') as f:
                f.write(classification_report_str)

            # Log classification report as artifact in MLflow
            mlflow.log_artifact(report_file_path)

            # Ensure the target directory exists
            os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)

            # Save metrics as a JSON file
            with open(metrics_filepath, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt ='d', cmap='Blues')
            plt.title(f'{model.__class__.__name__} Confusion Matrix ({stage})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()

            if stage=='baseline':
                save_path = os.path.join(self.model_base_dir, f'{model.__class__.__name__}_cm.png')
            # else:
            #     save_path = os.path.join(self.model_tuned_dir, f'{model.__class__.__name__}_cm.png')
            else:
                save_path = os.path.join(self.model_cm_dir, f'{model.__class__.__name__}_cm.png')
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()

            return metrics

        except Exception as e:
            logging.error("❌ Error laoding-splitting data")
            raise CustomException(str(e))
        
    
    def save_best_model(self, best_model_info):
        logging.info("Saving the model")
        
        # Extract model name and model from the best_model_info dictionary
        model_name = best_model_info['model_name']
        model = best_model_info['model']
        
        # Define the model saving path
        model_path = os.path.join(self.model_dir, f'{model_name}_best.pkl')
        
        # Ensure the directory exists before saving the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(model, model_path)
        logging.info(f'Best Model saved at {model_path}')



    def train_run(self):
        """Main method for running the model pipeline"""
        try:
            with mlflow.start_run():
                logging.info("Starting our Model Training pipeline")

                logging.info("Starting our MLFLOW experimentation")

                logging.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.val_path, artifact_path="datasets")

                # Step 1: Baseline models training + evaluation + logging
                X_train, y_train, X_val, y_val = self.load_split_data()
                best_lgbm_model, best_params = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_val, y_val)

                # Save the best model
                self.save_best_model({
                    'model_name': 'LGBM',  # Model name here
                    'model': best_lgbm_model
                })

                # Log the model to MLflow
                model_path = os.path.join(self.model_dir, 'LGBM_best.pkl')
                mlflow.log_artifact(model_path)

                # Log parameters and metrics to MLflow
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                # Log classification report to MLflow
                classification_report_str = metrics['class_report']
                report_file_path = os.path.join(self.model_metrics_dir, 'LGBM_class_report.txt')
                with open(report_file_path, 'w') as f:
                    f.write(classification_report_str)

                mlflow.log_artifact(report_file_path)

                logging.info("Model Training successfully completed")
        except Exception as e:
            logging.error("❌ Error during Model Training")
            raise CustomException(str(e))
        
    
if __name__=='__main__':
    logging.info('Step 3 - Model Training Pipeline')
    config = read_yaml(CONFIG_PATH)
    model_trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH_pre, PROCESSED_VAL_DATA_PATH_pre,
                                  MODEL_OUTPUT_PATH, config)
    model_trainer.train_run()
    
            