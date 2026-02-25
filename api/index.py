from http.server import BaseHTTPRequestHandler
import json
import pickle
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path so we can import anemia_pipeline
sys.path.append(os.path.dirname(__file__))
from anemia_pipeline import predict_anemia

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        try:
            # Extract inputs
            hemoglobin = float(data.get('hemoglobin'))
            mch = float(data.get('mch'))
            mchc = float(data.get('mchc'))
            mcv = float(data.get('mcv'))
            gender = int(data.get('gender'))
            
            # Optional inputs
            rbc = float(data.get('rbc', 0))
            age = float(data.get('age', 0))
            
            # The model file needs to be in a place where the script can find it.
            # In Vercel, the path might be relative to this file.
            
            result = predict_anemia(
                hemoglobin=hemoglobin,
                rbc=rbc,
                age=age,
                gender=gender,
                mcv=mcv,
                mch=mch,
                mchc=mchc
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        return
