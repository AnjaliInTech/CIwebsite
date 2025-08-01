from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import joblib
import os

# Create a Flask app instance
app = Flask(__name__)
# Set a secret key. This is required for flashing messages.
app.secret_key = 'your_super_secret_key_here'

# --- Model Loading ---
MODEL_PATH = 'RandomForest_pipeline.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please place the trained model file in the application's root directory.")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading the model file: {e}")
    print("Please ensure your scikit-learn, numpy, and scipy versions are compatible with the model.")
    exit(1)

# Define the features that your model expects
NUMERICAL_FEATURES = ['time_spent_alone', 'social_event_attendance', 'going_outside', 'friends_circle_size', 'post_frequency']
CATEGORICAL_FEATURES = ['stage_fear', 'drained_after_socializing']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Define the options for the categorical features
CATEGORICAL_OPTIONS = {
    'stage_fear': ['Yes', 'No'],
    'drained_after_socializing': ['Yes', 'No']
}

# --- Flask Routes ---
@app.route('/')
def home():
    """
    Renders the main page of the application with the new input form fields.
    """
    return render_template('index.html',
                           numerical_features=NUMERICAL_FEATURES,
                           categorical_features=CATEGORICAL_FEATURES,
                           categorical_options=CATEGORICAL_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission, processes the new input data,
    makes a prediction, and renders the result.
    """
    try:
        # Collect input data from the form
        input_data = {}

        # --- Handle Numerical Features ---
        for feature in NUMERICAL_FEATURES:
            value_str = request.form.get(feature)
            if not value_str:
                flash(f"Missing value for '{feature}'. Please provide all inputs.", 'error')
                return render_template('index.html',
                                       numerical_features=NUMERICAL_FEATURES,
                                       categorical_features=CATEGORICAL_FEATURES,
                                       categorical_options=CATEGORICAL_OPTIONS)
            try:
                input_data[feature] = float(value_str)
            except ValueError:
                flash(f"Invalid input for '{feature}'. Please enter a valid number.", 'error')
                return render_template('index.html',
                                       numerical_features=NUMERICAL_FEATURES,
                                       categorical_features=CATEGORICAL_FEATURES,
                                       categorical_options=CATEGORICAL_OPTIONS)

        # --- Handle Categorical Features ---
        for feature in CATEGORICAL_FEATURES:
            value = request.form.get(feature)
            if not value:
                flash(f"Missing value for '{feature}'. Please provide all inputs.", 'error')
                return render_template('index.html',
                                       numerical_features=NUMERICAL_FEATURES,
                                       categorical_features=CATEGORICAL_FEATURES,
                                       categorical_options=CATEGORICAL_OPTIONS)
            input_data[feature] = value

        # Create a pandas DataFrame from the new input data, ensuring columns are in the correct order
        input_df = pd.DataFrame([input_data], columns=ALL_FEATURES)

        # Make the prediction
        prediction_proba = model.predict_proba(input_df)[0]
        extrovert_proba = prediction_proba[1]

        if extrovert_proba > 0.5:
            prediction = 'Extrovert'
            confidence = extrovert_proba
        else:
            prediction = 'Introvert'
            confidence = 1 - extrovert_proba

        return render_template(
            'index.html',
            prediction=prediction,
            confidence=f"{confidence:.2%}",
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            categorical_options=CATEGORICAL_OPTIONS
        )

    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}. Please try again.", 'error')
        return render_template('index.html',
                               numerical_features=NUMERICAL_FEATURES,
                               categorical_features=CATEGORICAL_FEATURES,
                               categorical_options=CATEGORICAL_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True)
