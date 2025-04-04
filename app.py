from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]

    # Convert features to a numpy array and scale them
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    # Make prediction
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)[:, 1]

    # Map prediction to result
    result = "Positive (Heart Disease Detected)" if prediction[0] == 1 else "Negative (No Heart Disease)"

    # Return the result to the user
    return render_template('index.html', prediction_text=f'Prediction: {result}', probability=f'Probability: {prediction_proba[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)