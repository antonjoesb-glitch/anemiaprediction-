from http.server import BaseHTTPRequestHandler
import json
import pickle
import pandas as pd
import numpy as np
import os

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
        else:
            data = {}

        try:
            # Extract inputs
            hemoglobin = float(data.get('hemoglobin', 0))
            mch = float(data.get('mch', 0))
            mchc = float(data.get('mchc', 0))
            mcv = float(data.get('mcv', 1)) # avoid div by zero
            gender = int(data.get('gender', 0))
            
            hematocrit = float(hemoglobin) * 3 / 100
            est_rbc = (float(hematocrit) * 10) / float(mcv)
            
            input_data = {
                'Gender': gender,
                'Hemoglobin': hemoglobin,
                'MCH': mch,
                'MCHC': mchc,
                'MCV': mcv,
                'Hematocrit': hematocrit,
                'Estimated_RBC': est_rbc
            }
            
            # The model file is in the root directory, one level up from api/
            model_path = os.path.join(os.path.dirname(__file__), '..', 'anemia_prediction_model.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['features']
            
            input_df = pd.DataFrame([input_data])[features]
            input_scaled = scaler.transform(input_df)
            
            prob = model.predict_proba(input_scaled)[0, 1]
            pred = model.predict(input_scaled)[0]
            
            result = {
                'Anemia': 'Yes' if pred == 1 else 'No',
                'Probability': f"{prob:.2%}"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
