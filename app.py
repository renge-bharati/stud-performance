# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>Student Performance Predictor</title>
        <style>
            body { font-family: Arial; text-align: center; background-color: #f0f6ff; }
            .container { width: 400px; margin: 0 auto; padding: 20px; background: white; border-radius: 10px; box-shadow: 0px 0px 10px gray; }
            input, select { margin: 5px; padding: 10px; width: 90%; border-radius: 5px; border: 1px solid #ccc; }
            button { padding: 10px 20px; background-color: #007BFF; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎓 Student Performance Prediction</h2>
            <form action="/predict" method="post">
                <input type="number" name="hours_studied" placeholder="Hours Studied" required><br>
                <input type="number" name="attendance" placeholder="Attendance (%)" required><br>
                <input type="number" name="assignments_done" placeholder="Assignments Completed" required><br>
                <select name="extra_classes">
                    <option value="1">Takes Extra Classes</option>
                    <option value="0">No Extra Classes</option>
                </select><br>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours_studied = float(request.form['hours_studied'])
        attendance = float(request.form['attendance'])
        assignments_done = float(request.form['assignments_done'])
        extra_classes = int(request.form['extra_classes'])

        features = np.array([[hours_studied, attendance, assignments_done, extra_classes]])
        prediction = model.predict(features)

        return f"<h2 style='text-align:center;'>Predicted Score: {round(prediction[0], 2)}</h2>"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
