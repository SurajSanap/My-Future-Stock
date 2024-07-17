from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models
with open('knn_classification_model.pkl', 'rb') as f:
    knn_classification_model = pickle.load(f)

with open('knn_regression_model.pkl', 'rb') as f:
    knn_regression_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input values to float
    try:
        open_close = float(data['Open - Close'])
        high_low = float(data['High - Low'])
    except ValueError as e:
        return jsonify({"error": "Invalid input data"}), 400
    
    features = np.array([[open_close, high_low]])
    
    classification_prediction = knn_classification_model.predict(features)
    regression_prediction = knn_regression_model.predict(features)
    
    return jsonify({
        'Should Buy Stock': 'Yes' if classification_prediction[0] == 1 else 'No',
        'Predicted Stock Price': f'{regression_prediction[0]:.2f}'
    })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
