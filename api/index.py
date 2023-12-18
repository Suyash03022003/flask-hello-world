from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from joblib import load

app = Flask(__name__)
CORS(app)

model = load('nursery.joblib')

def get_class_name(class_names):
    if class_names == 0:
        return "High Learner"
    elif class_names == 1:
        return "Medium Learner"
    elif class_names == 2:
        return "Slow Learner"
    
def parse_text_data(text_data):
    data = []

    for line in text_data.split('\n'):
        line = line.strip()
        if line:
            cleaned_line = line.replace('[', '').replace(']', '').replace(' ', '')

            values = list(map(float, cleaned_line.split(',')))
            data.append(values)
    return data

def predict(data):
    input_data = pd.DataFrame(data) 
    predictions = model.predict(input_data)
    return predictions.tolist()

@app.route('/predict', methods=['POST'])
def make_predictions():
    try:
        text_data = request.get_data(as_text=True) 
        data = parse_text_data(text_data)
        print("Parsed Data:", data) 
        predictions = predict(data)
        class_names = [get_class_name(prediction) for prediction in predictions]
        return jsonify({'predictions': class_names})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
