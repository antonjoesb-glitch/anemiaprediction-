from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import re
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

PATTERNS = {
    'hemoglobin': r'(?:hemo?globin|hgb|hb)[^\d]+([\d]+\.?[\d]*)',
    'mcv':        r'\bmcv\b[^\d]+([\d]+\.?[\d]*)',
    'mch':        r'\bmch\b[^\d]+([\d]+\.?[\d]*)',
    'mchc':       r'\bmchc\b[^\d]+([\d]+\.?[\d]*)',
    'rbc':        r'\brbc\b[^\d]+([\d]+\.?[\d]*)',
    'age':        r'\bage\s*[:\s]*(\d+)',
}

def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

import requests

def extract_text_from_image(file_bytes):
    try:
        # Try OCR.space free API first (works natively on Vercel without Tesseract binary)
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': ('image.jpg', file_bytes, 'image/jpeg')},
            data={'apikey': 'helloworld', 'language': 'eng'}
        )
        result = response.json()
        if result.get('IsErroredOnProcessing') or not result.get('ParsedResults'):
            raise Exception("OCR API Failed or returned no results")
        
        text = ""
        for item in result.get('ParsedResults', []):
            text += item.get('ParsedText', '') + "\n"
        return text
    except Exception as api_err:
        try:
            # Fallback to local pytesseract (if running locally and Tesseract is installed)
            image = Image.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(image)
        except Exception:
            raise Exception("Image OCR is not supported natively in this cloud environment. Please upload a PDF report instead.")

def parse_values(text):
    text = text.lower()
    values = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            try:
                values[key] = float(match.group(1))
            except ValueError:
                pass
                
    gender_match = re.search(r'\b(?:sex|gender)\s*[:\s]*\b(male|female)\b', text)
    if gender_match:
        if gender_match.group(1) == 'male':
            values['gender'] = 1
        else:
            values['gender'] = 0
            
    return values

import pickle
import pandas as pd

def predict_anemia_inline(hemoglobin, rbc, age, gender, mcv, mch, mchc, hematocrit=None):
    try:
        with open('anemia_prediction_model.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return {"error": "Model file not found. Please run the pipeline first."}
    
    model = data['model']
    scaler = data['scaler']
    features = data['features']
    
    if hematocrit is None or hematocrit == 0:
        hematocrit = float(hemoglobin) * 3 / 100
    
    est_rbc = (float(hematocrit) * 10) / float(mcv) if mcv else 0
    
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

@app.route('/', defaults={'path': ''}, methods=['POST', 'GET'])
@app.route('/<path:path>', methods=['POST', 'GET'])
def extract_report(path):
    if request.method == 'GET':
        return jsonify({"status": "API is running. Send POST with a file."})
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        file_bytes = file.read()
        filename_lower = file.filename.lower()
        
        text = ""
        if filename_lower.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_bytes)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        extracted_values = parse_values(text)
        
        # Gender and Age might be POST fields that are optionally sent
        gender = int(request.form.get('gender') or extracted_values.get('gender', 0))
        age = float(request.form.get('age') or extracted_values.get('age', 25))
        
        # Now call the model to get probability, using defaults for missing report values
        hemoglobin = float(extracted_values.get('hemoglobin', 0))
        mch = float(extracted_values.get('mch', 0))
        mchc = float(extracted_values.get('mchc', 0))
        mcv = float(extracted_values.get('mcv', 1))
        rbc = float(extracted_values.get('rbc', 0))
        
        try:
            result = predict_anemia_inline(
                hemoglobin=hemoglobin,
                rbc=rbc,
                age=age,
                gender=gender,
                mcv=mcv,
                mch=mch,
                mchc=mchc
            )
            extracted_values['prediction'] = result
        except Exception as e:
            extracted_values['prediction_error'] = str(e)
            
        return jsonify({
            'success': True,
            'extracted_values': extracted_values,
            'raw_text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8735)
