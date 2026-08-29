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
    
    # Generate clinical explanation based on WHO limits
    prob_percentage = prob * 100
    explanations = []
    
    hb_normal_min = 13.5 if int(gender) == 1 else 12.0
    is_anemic_by_who = hemoglobin < hb_normal_min
    
    if hemoglobin > 0:
        if is_anemic_by_who:
            explanations.append(f"Your Hemoglobin level is {hemoglobin:.1f} g/dL, which is below the healthy threshold ({hb_normal_min} g/dL for your gender). This is the primary indicator of anemia.")
        elif prob_percentage > 40:
            explanations.append(f"Your Hemoglobin level ({hemoglobin:.1f} g/dL) is normal, but other red blood cell indices show anomalies.")
            
        if mcv > 0:
            if mcv < 80:
                explanations.append(f"Your MCV is {mcv:.1f} fL, which is abnormally small (Microcytic). This strongly suggests iron deficiency.")
            elif mcv > 100:
                explanations.append(f"Your MCV is {mcv:.1f} fL, which is abnormally large (Macrocytic). This often points to Vitamin B12 or Folate deficiency.")
                
        if mch > 0 and mch < 27:
            explanations.append(f"Your MCH is low at {mch:.1f} pg, meaning your red blood cells carry less hemoglobin than normal (hypochromic).")
            
    if not explanations:
        if prob_percentage < 30:
            explanation_text = "All key Complete Blood Count (CBC) metrics are within healthy reference ranges. Maintain a balanced, iron-rich diet to keep these levels optimal."
        else:
            explanation_text = "While primary metrics appear normal, subtle multi-variable correlations in your blood indices (such as MCHC and RBC count) slightly elevated your algorithmic risk score."
    else:
        explanation_text = " ".join(explanations)
        if is_anemic_by_who or prob_percentage > 50:
            explanation_text += " Recommendation: Please consult a physician for a clinical diagnosis and potential dietary or supplement interventions."

    return {
        'Anemia': 'Yes' if pred == 1 else 'No',
        'Probability': f"{prob:.2%}",
        'Explanation': explanation_text
    }
app = Flask(__name__)
CORS(app)

@app.route('/', defaults={'path': ''}, methods=['POST', 'GET'])
@app.route('/<path:path>', methods=['POST', 'GET'])
def predict(path):
    if request.method == 'GET':
        return jsonify({"status": "API is running. Send POST with data."})
    try:
        data = request.json or {}
        # Extract inputs
        hemoglobin = float(data.get('hemoglobin') or 0)
        mch = float(data.get('mch') or 0)
        mchc = float(data.get('mchc') or 0)
        mcv = float(data.get('mcv') or 1) # avoid div by zero
        gender = int(data.get('gender') or 0)
        
        # Optional inputs that might not be used by the current model but requested in UI
        rbc = float(data.get('rbc') or 0)
        age = float(data.get('age') or 0)
        hematocrit_val = data.get('hematocrit')
        hematocrit = float(hematocrit_val) if hematocrit_val else None
        
        # Get prediction
        result = predict_anemia(
            hemoglobin=hemoglobin,
            rbc=rbc,
            age=age,
            gender=gender,
            mcv=mcv,
            mch=mch,
            mchc=mchc,
            hematocrit=hematocrit
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8734)
