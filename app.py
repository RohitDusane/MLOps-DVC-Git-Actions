from flask import Flask, render_template, request
import os
import numpy as np
import joblib  # for loading the model
from src.CreditRisk.pipeline.training import main as training_pipeline
from src.CreditRisk.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Load the model once, during application startup
MODEL_PATH = "artifacts/models/LGBM_best.pkl"
model = None

def load_model():
    """ Load model if it exists. """
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None

# Initial model load on startup
load_model()

@app.route('/', methods=['GET'])  # Route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # Route to train the model
def trainModel():
    try:
        # Run training pipeline, assuming this will save the model
        training_pipeline()  # This triggers the pipeline and trains the model
        load_model()  # Reload the model after training
        return "Training Successful!"
    except Exception as e:
        print(f"Training failed: {e}")
        return "There was an error in training the model."

@app.route('/predict', methods=['POST', 'GET'])  # Route to make predictions
def predict():
    if request.method == 'POST':
        try:
            # Read inputs from the user
            credit_score = float(request.form['credit_score'])
            credit_limit_used = float(request.form['credit_limit_used'])
            prev_defaults = int(request.form['prev_defaults'])
            default_in_last_6months = int(request.form['default_in_last_6months'])
            no_of_days_employed = float(request.form['no_of_days_employed'])
            yearly_debt_payments = float(request.form['yearly_debt_payments'])
            age = int(request.form['age'])
            net_yearly_income = float(request.form['net_yearly_income'])
            credit_limit = float(request.form['credit_limit'])
            owns_car = int(request.form['owns_car'])  # 0 for No, 1 for Yes

            # Prepare the input data as a list
            data = [
                credit_score, credit_limit_used, prev_defaults, default_in_last_6months,
                no_of_days_employed, yearly_debt_payments, age, net_yearly_income,
                credit_limit, owns_car
            ]

            # Convert to numpy array (as expected by the model)
            data = np.array(data).reshape(1, -1)  # Reshape to 2D array for prediction

            # Instantiate the prediction pipeline and make predictions
            obj = PredictionPipeline()
            prediction = obj.predict(data)
            prediction_proba = obj.predict_proba(data)

            # Determine risk status (Risk or No Risk)
            if prediction[0] == 1:
                risk_status = "Risk"
            else:
                risk_status = "No Risk"

            # Get the probability of the 'Risk' class (class 1)
            risk_probability = prediction_proba[0][1]  # Probability for class 1 (Risk)

            # Format the probability to 2 decimal places
            risk_probability = round(risk_probability, 2)

            # Return the result with the risk status and the formatted probability
            return render_template('results.html', prediction=risk_status, probability=risk_probability)

        except Exception as e:
            print('Error:', e)
            return 'There was an error processing your request'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)