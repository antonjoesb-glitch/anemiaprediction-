from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

def predict_anemia(hemoglobin, rbc, age, gender, mcv, mch, mchc, hematocrit=None):
    try:
        with open('anemia_prediction_model.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return "Model file not found. Please run the pipeline first."
    
    model = data['model']
    scaler = data['scaler']
    features = data['features']
    
    if hematocrit is None:
        hematocrit = float(hemoglobin) * 3 / 100
    
    est_rbc = (float(hematocrit) * 10) / float(mcv)
    
    input_data = {
        'Gender': int(gender),
        'Hemoglobin': float(hemoglobin),
        'MCH': float(mch),
        'MCHC': float(mchc),
        'MCV': float(mcv),
        'Hematocrit': float(hematocrit),
        'Estimated_RBC': est_rbc
    }
    
    input_df = pd.DataFrame([input_data])[features]
    input_scaled = scaler.transform(input_df)
    
    prob = model.predict_proba(input_scaled)[0, 1]
    pred = model.predict(input_scaled)[0]
    
    return {
        'Anemia': 'Yes' if pred == 1 else 'No',
        'Probability': f"{prob:.2%}"
    }
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract inputs
        hemoglobin = float(data.get('hemoglobin'))
        mch = float(data.get('mch'))
        mchc = float(data.get('mchc'))
        mcv = float(data.get('mcv'))
        gender = int(data.get('gender'))
        
        # Optional inputs that might not be used by the current model but requested in UI
        rbc = float(data.get('rbc', 0))
        age = float(data.get('age', 0))
        
        # Get prediction
        result = predict_anemia(
            hemoglobin=hemoglobin,
            rbc=rbc,
            age=age,
            gender=gender,
            mcv=mcv,
            mch=mch,
            mchc=mchc
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8734)
