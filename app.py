from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
knn_classification_model = pickle.load(open('knn_classification_model.pkl', 'rb'))
knn_regression_model = pickle.load(open('knn_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        open_close = float(request.form['open_close'])
        high_low = float(request.form['high_low'])
        
        features = np.array([[open_close, high_low]])
        
        classification_prediction = knn_classification_model.predict(features)
        regression_prediction = knn_regression_model.predict(features)
        
        return render_template('index.html', 
                               classification_result=classification_prediction[0], 
                               regression_result=regression_prediction[0])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
