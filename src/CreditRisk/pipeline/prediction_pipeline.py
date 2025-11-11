import pandas as pd
import joblib
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Load the trained model from the specified path
        self.model = joblib.load(Path('artifacts/models/LGBM_best.pkl'))

        # You should define or load the feature names that were used during training
        self.features = [
            'credit_score', 'credit_limit_used', 'prev_defaults', 
            'default_in_last_6months', 'no_of_days_employed', 
            'yearly_debt_payments', 'age', 'net_yearly_income', 
            'credit_limit', 'owns_car'
        ]

    def predict(self, data):
        # Convert the input data to a pandas DataFrame with the same feature names
        data_df = pd.DataFrame(data, columns=self.features)
        prediction = self.model.predict(data_df)
        return prediction

    def predict_proba(self, data):
        # Convert the input data to a pandas DataFrame with the same feature names
        data_df = pd.DataFrame(data, columns=self.features)
        prediction_proba = self.model.predict_proba(data_df)
        return prediction_proba
