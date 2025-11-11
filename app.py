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






# from flask import Flask, render_template, request
# import os
# import numpy as np
# from src.CreditRisk.pipeline import training  # Assuming you have a pipeline set up

# app = Flask(__name__)  # initializing a Flask app

# @app.route('/', methods=['GET'])  # route to display the home page
# def homePage():
#     return render_template("index.html")

# @app.route('/train', methods=['GET'])  # route to train the model
# def training():
#     os.system("python src.CreditRisk.pipeline.training.py")  # run your training script
#     return "Training Successful!" 

# @app.route('/predict', methods=['POST', 'GET'])  # route to make predictions
# def index():
#     if request.method == 'POST':
#         try:
#             # Reading the inputs given by the user
#             credit_score = float(request.form['credit_score'])
#             credit_limit_used = float(request.form['credit_limit_used'])
#             prev_defaults = int(request.form['prev_defaults'])
#             default_in_last_6months = int(request.form['default_in_last_6months'])
#             no_of_days_employed = float(request.form['no_of_days_employed'])
#             yearly_debt_payments = float(request.form['yearly_debt_payments'])
#             age = int(request.form['age'])
#             net_yearly_income = float(request.form['net_yearly_income'])
#             credit_limit = float(request.form['credit_limit'])
#             owns_car = int(request.form['owns_car'])  # 0 for No, 1 for Yes

#             # Prepare data in the right shape for prediction
#             data = [credit_score, credit_limit_used, prev_defaults, default_in_last_6months,
#                     no_of_days_employed, yearly_debt_payments, age, net_yearly_income, 
#                     credit_limit, owns_car]
#             data = np.array(data).reshape(1, -1)

#             # Initialize the prediction pipeline
#             obj = training()
#             prediction = obj.predict(data)

#             return render_template('results.html', prediction=str(prediction))

#         except Exception as e:
#             print('Error:', e)
#             return 'There was an error processing your request'

#     else:
#         return render_template('index.html')

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)





















# import os
# import joblib
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify, render_template

# # ---- Create Flask App Instance ----
# app = Flask(__name__)

# # ---- Define Paths ----
# MODEL_PATH = "artifacts/models/LGBM_best.pkl"
# CAT_IMPUTER_PATH = "artifacts/prepro/cat_imputer.pkl"
# NUM_IMPUTER_PATH = "artifacts/prepro/num_imputer.pkl"
# SCALER_PATH = "artifacts/prepro/numerical_scaler.pkl"
# LABEL_ENCODERS_PATH = {
#     # "credit_card_default": "artifacts/prepro/credit_card_default_label_encoder.pkl",
#     # "gender": "artifacts/prepro/gender_label_encoder.pkl",
#     # "occupation_type": "artifacts/prepro/occupation_type_label_encoder.pkl",
#     "owns_car": "artifacts/prepro/owns_car_label_encoder.pkl",
#     # "owns_house": "artifacts/prepro/owns_house_label_encoder.pkl"
# }

# # ---- Load Model and Preprocessors ----
# try:
#     model = joblib.load(MODEL_PATH)
#     print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    
#     # Load preprocessing objects
#     cat_imputer = joblib.load(CAT_IMPUTER_PATH)
#     num_imputer = joblib.load(NUM_IMPUTER_PATH)
#     numerical_scaler = joblib.load(SCALER_PATH)
    
#     label_encoders = {}
#     for feature, path in LABEL_ENCODERS_PATH.items():
#         label_encoders[feature] = joblib.load(path)
    
# except Exception as e:
#     print(f"❌ Failed to load model or preprocessing objects. Error: {str(e)}")
#     model = None
#     cat_imputer = None
#     num_imputer = None
#     numerical_scaler = None
#     label_encoders = None


# # ======================================================
# # ROUTE 1: Home Page (HTML Form)
# # ======================================================
# @app.route("/")
# def home():
#     """
#     Render a simple HTML form for manual input.
#     """
#     return render_template("index.html")


# # ======================================================
# # ROUTE 2: Predict via JSON API
# # ======================================================
# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Make a credit card default prediction.
#     Accepts JSON input containing feature values.
#     Returns default prediction and probability.
#     """
#     if model is None:
#         return jsonify({"error": "Model not loaded"}), 500

#     try:
#         # ---- Parse JSON input ----
#         data = request.get_json()

#         # Convert input data to DataFrame
#         input_df = pd.DataFrame([data])

#         # ---- Preprocessing (Imputation, Encoding, Scaling) ----
#         # ---- Preprocessing ----
#         # Apply categorical imputations
#         cat_columns = ['gender', 'occupation_type', 'owns_car']  # Adjust based on your features
#         input_df[cat_columns] = cat_imputer.transform(input_df[cat_columns])

#         # Apply numerical imputations
#         num_columns = ["credit_score", "credit_limit_used", "prev_defaults", "default_in_last_6months", 
#                     "no_of_days_employed", "yearly_debt_payments", "age", "net_yearly_income", "credit_limit"]  # All 10 features
#         input_df[num_columns] = num_imputer.transform(input_df[num_columns])

#         # Apply label encoding
#         for col in ['gender', 'occupation_type', 'owns_car']:  # All relevant categorical columns
#             input_df[col] = label_encoders[col].transform(input_df[col])

#         # Apply numerical scaling
#         input_df[num_columns] = numerical_scaler.transform(input_df[num_columns])

#         # # Impute missing values
#         # cat_columns = ['gender', 'occupation_type', 'owns_car', 'owns_house']  # categorical features
#         # for col in cat_columns:
#         #     input_df[col] = cat_imputer.transform(input_df[col])

#         # # Apply label encoding for categorical columns
#         # for col in label_encoders.keys():
#         #     input_df[col] = label_encoders[col].transform(input_df[col])

#         # # Impute and scale numerical features
#         # num_columns = ['credit_score', 'credit_limit_used', 'prev_defaults', 'no_of_days_employed', 
#         #                'yearly_debt_payments', 'age', 'net_yearly_income', 'credit_limit']
#         # input_df[num_columns] = num_imputer.transform(input_df[num_columns])
#         # input_df[num_columns] = numerical_scaler.transform(input_df[num_columns])

#         # ---- Make Prediction ----
#         pred_class = model.predict(input_df)[0]
#         pred_prob = model.predict_proba(input_df)[0][1]  # probability of default = class 1

#         # ---- Build Response ----
#         response = {
#             "prediction": int(pred_class),
#             "default_probability": round(float(pred_prob), 4),
#             "status": "success"
#         }

#         return jsonify(response)

#     except Exception as e:
#         return jsonify({
#             "error": str(e),
#             "message": "Failed to generate prediction"
#         }), 400


# # ======================================================
# # ROUTE 3: Predict via HTML Form
# # ======================================================
# # ======================================================
# # ROUTE 3: Predict via HTML Form
# # ======================================================
# @app.route("/predict_form", methods=["POST"])
# def predict_form():
#     """
#     Handle prediction via HTML form submission.
#     """
#     try:
#         # Get input values from form
#         form_values = [float(x) for x in request.form.values()]

#         # Convert to DataFrame
#         feature_names = [
#             "credit_score", "credit_limit_used", "prev_defaults", "default_in_last_6months", 
#             "no_of_days_employed", "yearly_debt_payments", "age", "net_yearly_income", 
#             "credit_limit", "owns_car"
#         ]  # updated feature names
#         input_df = pd.DataFrame([form_values], columns=feature_names)

#         # ---- Preprocessing: Apply the same transformations as during training ----

#         # ---- Preprocessing ----
#         # Apply categorical imputations
#         cat_columns = ['gender', 'occupation_type', 'owns_car']  # Adjust based on your features
#         input_df[cat_columns] = cat_imputer.transform(input_df[cat_columns])

#         # Apply numerical imputations
#         num_columns = ["credit_score", "credit_limit_used", "prev_defaults", "default_in_last_6months", 
#                     "no_of_days_employed", "yearly_debt_payments", "age", "net_yearly_income", "credit_limit"]  # All 10 features
#         input_df[num_columns] = num_imputer.transform(input_df[num_columns])

#         # Apply label encoding
#         for col in ['gender', 'occupation_type', 'owns_car']:  # All relevant categorical columns
#             input_df[col] = label_encoders[col].transform(input_df[col])

#         # Apply numerical scaling
#         input_df[num_columns] = numerical_scaler.transform(input_df[num_columns])


#         # ---- Make Prediction ----
#         pred_class = model.predict(input_df)[0]
#         pred_prob = model.predict_proba(input_df)[0][1]  # probability of default = class 1

#         return render_template(
#             "index.html",
#             prediction_text=f"Predicted Default Risk: {pred_class} (Probability: {round(pred_prob, 4)})"
#         )

#     except Exception as e:
#         return render_template("index.html", prediction_text=f"Error: {str(e)}")


# # ======================================================
# # RUN FLASK APP
# # ======================================================
# if __name__ == "__main__":
#     # Run Flask in debug mode
#     app.run(host="0.0.0.0", port=5000, debug=True)












# import joblib
# import pandas as pd
# from src.CreditRisk.logger import logging
# from flask import Flask, render_template, request, send_file,jsonify
# from io import StringIO
# from pathlib import Path

# app = Flask(__name__)

# # Load model and transformations
# MODEL_PATH = Path('artifacts/models/LGBM_best.pkl')
# model = joblib.load(MODEL_PATH)

# # Load transformations
# cat_imputer = joblib.load('artifacts/processed/transformations/cat_imputer.pkl')
# credit_card_default_label_encoder = joblib.load('artifacts/processed/transformations/credit_card_default_label_encoder.pkl')
# gender_label_encoder = joblib.load('artifacts/processed/transformations/gender_label_encoder.pkl')
# num_imputer = joblib.load('artifacts/processed/transformations/num_imputer.pkl')
# numerical_scaler = joblib.load('artifacts/processed/transformations/numerical_scaler.pkl')
# occupation_type_label_encoder = joblib.load('artifacts/processed/transformations/occupation_type_label_encoder.pkl')
# owns_car_label_encoder = joblib.load('artifacts/processed/transformations/owns_car_label_encoder.pkl')
# owns_house_label_encoder = joblib.load('artifacts/processed/transformations/owns_house_label_encoder.pkl')
# random_forest_model = joblib.load('artifacts/processed/transformations/random_forest_model.pkl')
# skewed_columns = joblib.load('artifacts/processed/transformations/skewed_columns.pkl')

# def apply_transformations(df):

#     if 'credit_card_default' in df.columns:
#             df = df.drop(columns=['credit_card_default'], axis=1)
#     else:
#         logging.warning("Column 'credit_card_default' not found in the dataframe.")

#     if df.empty:
#         print("Input DataFrame is empty. Skipping transformations.")
#         return df  # Return as-is if it's empty
    
#     # Check for categorical columns
#     categorical_columns = df.select_dtypes(include=['object']).columns
#     print("Categorical columns:", categorical_columns)
    
#     if len(categorical_columns) > 0:
#         # Apply imputer only if there are categorical columns
#         df[categorical_columns] = cat_imputer.transform(df[categorical_columns])
#     else:
#         print("No categorical columns to impute.")

#     # Label encoding for categorical columns
#     df['credit_card_default'] = credit_card_default_label_encoder.transform(df['credit_card_default'])
#     df['gender'] = gender_label_encoder.transform(df['gender'])
#     df['occupation_type'] = occupation_type_label_encoder.transform(df['occupation_type'])
#     df['owns_car'] = owns_car_label_encoder.transform(df['owns_car'])
#     df['owns_house'] = owns_house_label_encoder.transform(df['owns_house'])

#     # Handle numerical missing values (imputation)
#     numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
#     df[numerical_columns] = num_imputer.transform(df[numerical_columns])

#     # Scale numerical features
#     df[numerical_columns] = numerical_scaler.transform(df[numerical_columns])

#     return df

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         prediction_type = request.form.get('prediction_type')
        
#         if prediction_type == 'single':
#             return render_template('single_prediction.html')
#         elif prediction_type == 'batch':
#             return render_template('batch_prediction.html')
    
#     return render_template('index.html')

# @app.route('/predict_single', methods=['POST'])
# def predict_single():
#     try:
#         # Get form data
#         data = {
#             'credit_score': float(request.form.get('credit_score', 0)),
#             'credit_limit_used': float(request.form.get('credit_limit_used', 0)),
#             'prev_defaults': float(request.form.get('prev_defaults', 0)),
#             'default_in_last_6months': float(request.form.get('default_in_last_6months', 0)),
#             'no_of_days_employed': float(request.form.get('no_of_days_employed', 0)),
#             'yearly_debt_payments': float(request.form.get('yearly_debt_payments', 0)),
#             'age': float(request.form.get('age', 0)),
#             'net_yearly_income': float(request.form.get('net_yearly_income', 0)),
#             'credit_limit': float(request.form.get('credit_limit', 0)),
#             'owns_car': float(request.form.get('owns_car', 0))
#         }

#         # Create a DataFrame from the form data
#         features_df = pd.DataFrame([data])

#         # Apply transformations (e.g., scaling, encoding)
#         features_df = apply_transformations(features_df)
        
#         # Make predictions
#         prediction = model.predict(features_df)  # Assuming you have a trained model

#         # Return the prediction (e.g., whether the person is likely to default)
#         return jsonify({'prediction': prediction.tolist()})

#     except ValueError:
#         return "Invalid input, please check your entries."
    
#     # Log the data to ensure it's correct
#     print("Form data:", data)

#     # Create DataFrame (use columns explicitly to avoid empty DataFrame)
#     columns = [
#         'credit_score', 'credit_limit_used', 'prev_defaults', 'default_in_last_6months',
#         'no_of_days_employed', 'yearly_debt_payments', 'age', 'net_yearly_income', 
#         'credit_limit', 'owns_car'
#     ]
#     features_df = pd.DataFrame([data], columns=columns)
    
#     print("Features DataFrame:", features_df)  # Check the DataFrame
#     # Apply transformations
#     features_df = apply_transformations(features_df)

#     # Make prediction
#     prediction = model.predict(features_df)
#     probability = model.predict_proba(features_df)[:, 1][0]  # Probability of class 1 (Risk)
    
#     # Determine risk classification
#     risk_class = "Risk" if probability >= 0.2 else "No Risk"
    
#     # Return results as part of the response to be rendered in results.html
#     return render_template('results.html', prediction=risk_class, probability=probability * 100)


# @app.route('/predict_batch', methods=['POST'])
# def predict_batch():
#     # Handle CSV file upload
#     file = request.files['csv_file']
#     df = pd.read_csv(file)

#     # Apply transformations to the batch data
#     df = apply_transformations(df)

#     # Make predictions
#     predictions = model.predict(df)
#     probabilities = model.predict_proba(df)[:, 1]  # Risk probability

#     # Add predictions and probabilities to dataframe
#     df['Prediction'] = predictions
#     df['Probability'] = probabilities

#     # Save result to CSV
#     result_csv = StringIO()
#     df.to_csv(result_csv, index=False)
#     result_csv.seek(0)
    
#     return send_file(result_csv, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)
