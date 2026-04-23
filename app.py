from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and features
model = joblib.load('fraud_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
engine = create_engine('sqlite:///insurance_claims.db')

def preprocess_input(data):
    from sklearn.preprocessing import LabelEncoder
    df = pd.DataFrame([data])
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    # Align with training features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

@app.route('/')
def home():
    # Get stats from database
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM claims")).scalar()
        fraud = conn.execute(text("SELECT COUNT(*) FROM claims WHERE fraud_reported='Y'")).scalar()
        avg_claim = conn.execute(text("SELECT ROUND(AVG(total_claim_amount),2) FROM claims")).scalar()
        recent = conn.execute(text("""
            SELECT age, insured_sex, incident_type, total_claim_amount, fraud_reported 
            FROM claims 
            ORDER BY ROWID DESC 
            LIMIT 5
        """)).fetchall()
    return render_template('index.html', 
                         total=total, fraud=fraud, 
                         avg_claim=avg_claim, recent=recent)

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'age': int(request.form.get('age', 35)),
        'months_as_customer': int(request.form.get('months_as_customer', 12)),
        'policy_deductable': int(request.form.get('policy_deductable', 1000)),
        'policy_annual_premium': float(request.form.get('policy_annual_premium', 1200)),
        'umbrella_limit': int(request.form.get('umbrella_limit', 0)),
        'insured_sex': request.form.get('insured_sex', 'MALE'),
        'insured_education_level': request.form.get('insured_education_level', 'Bachelor'),
        'insured_occupation': request.form.get('insured_occupation', 'craft-repair'),
        'insured_hobbies': request.form.get('insured_hobbies', 'sleeping'),
        'insured_relationship': request.form.get('insured_relationship', 'husband'),
        'incident_type': request.form.get('incident_type', 'Single Vehicle Collision'),
        'collision_type': request.form.get('collision_type', 'Side Collision'),
        'incident_severity': request.form.get('incident_severity', 'Minor Damage'),
        'authorities_contacted': request.form.get('authorities_contacted', 'Police'),
        'incident_hour_of_the_day': int(request.form.get('incident_hour', 12)),
        'number_of_vehicles_involved': int(request.form.get('num_vehicles', 1)),
        'bodily_injuries': int(request.form.get('bodily_injuries', 0)),
        'witnesses': int(request.form.get('witnesses', 1)),
        'police_report_available': request.form.get('police_report', 'YES'),
        'total_claim_amount': int(request.form.get('total_claim_amount', 5000)),
        'injury_claim': int(request.form.get('injury_claim', 1000)),
        'property_claim': int(request.form.get('property_claim', 2000)),
        'vehicle_claim': int(request.form.get('vehicle_claim', 2000)),
        'auto_make': request.form.get('auto_make', 'Toyota'),
        'auto_year': int(request.form.get('auto_year', 2015)),
        'property_damage': request.form.get('property_damage', 'YES'),
        'insured_zip': 430632,
        'capital-gains': 0,
        'capital-loss': 0,
        'policy_csl': '250/500',
        'policy_state': 'OH',
        'incident_state': 'OH',
        'incident_city': 'Columbus',
        'auto_model': 'Camry',
    }

    processed = preprocess_input(data)
    prob = model.predict_proba(processed)[0][1]
    prediction = 'FRAUD' if prob >= 0.3 else 'LEGITIMATE'

    return jsonify({
        'prediction': prediction,
        'confidence': round(prob * 100, 2),
        'risk_level': 'HIGH' if prob >= 0.6 else 'MEDIUM' if prob >= 0.3 else 'LOW'
    })

if __name__ == '__main__':
    app.run(debug=True)