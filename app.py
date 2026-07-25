from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from anemia_pipeline import predict_anemia

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
