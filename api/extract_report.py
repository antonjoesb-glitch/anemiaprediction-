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

def extract_text_from_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise Exception("Image OCR is not supported in this cloud environment. Please upload a PDF report instead.")

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
            from predict import predict_anemia
            result = predict_anemia(
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
