from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb_lib

app = Flask(__name__)

# Load models
rf = joblib.load('random_forest.pkl')
iso = joblib.load('isolation_forest.pkl')
encoders = joblib.load('label_encoders.pkl')
# xgb = joblib.load('xgboost.pkl')
meta_model = joblib.load('meta_model.pkl')
xgb = xgb_lib.XGBClassifier()
xgb.load_model('xgboost.json')

# Build dropdown options from encoders
CATEGORIES = list(encoders['category'].classes_)
GENDERS = list(encoders['gender'].classes_)
CITIES = list(encoders['city'].classes_)
STATES = list(encoders['state'].classes_)
JOBS = list(encoders['job'].classes_)

def encode_input(col, value):
    return int(encoders[col].transform([value])[0])

def get_confidence_score(rf_prob, xgb_prob, iso_pred):
    # Meta model takes RF and XGBoost probs as input
    meta_input = np.array([[rf_prob, xgb_prob]])
    meta_prob = meta_model.predict_proba(meta_input)[0][1]
    
    # Isolation Forest adds anomaly weight
    iso_weight = 0.1 if iso_pred == -1 else 0
    final_score = (meta_prob * 0.9) + iso_weight
    
    if final_score < 0.4:
        label = "Safe"
    elif final_score < 0.7:
        label = "Suspicious"
    else:
        label = "Fraud"
    return round(final_score * 100, 2), label

@app.route('/')
def home():
    return render_template('index.html',
        categories=CATEGORIES,
        genders=GENDERS,
        cities=CITIES,
        states=STATES,
        jobs=JOBS
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    try:
        category = encode_input('category', data['category'])
        gender = encode_input('gender', data['gender'])
        city = encode_input('city', data['city'])
        state = encode_input('state', data['state'])
        job = encode_input('job', data['job'])
        
        lat = float(data['lat'])
        long = float(data['long'])
        merch_lat = float(data['merch_lat'])
        merch_long = float(data['merch_long'])
        distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)

        features = np.array([[
            category, float(data['amt']), gender,
            city, state, int(data['zip']),
            lat, long, int(data['city_pop']),
            job, merch_lat, merch_long,
            int(data['hour']), int(data['day_of_week']),
            int(data['age']), distance
        ]])

        rf_prob = rf.predict_proba(features)[0][1]
        xgb_prob = xgb.predict_proba(features)[0][1]
        iso_pred = iso.predict(features)[0]
        
        score, label = get_confidence_score(rf_prob, xgb_prob, iso_pred)
        return jsonify({'score': score, 'label': label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'GET':
        return render_template('batch.html')
    
    file = request.files['file']
    df = pd.read_csv(file)
    
    results = []
    for _, row in df.iterrows():
        try:
            category = encode_input('category', row['category'])
            gender = encode_input('gender', row['gender'])
            city = encode_input('city', row['city'])
            state = encode_input('state', row['state'])
            job = encode_input('job', row['job'])
            
            lat = float(row['lat'])
            long = float(row['long'])
            merch_lat = float(row['merch_lat'])
            merch_long = float(row['merch_long'])
            distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)

            features = np.array([[
                category, float(row['amt']), gender,
                city, state, int(row['zip']),
                lat, long, int(row['city_pop']),
                job, merch_lat, merch_long,
                int(row['hour']), int(row['day_of_week']),
                int(row['age']), distance
            ]])

            rf_prob = rf.predict_proba(features)[0][1]
            xgb_prob = xgb.predict_proba(features)[0][1]
            iso_pred = iso.predict(features)[0]
            score, label = get_confidence_score(rf_prob, xgb_prob, iso_pred)
            results.append({
                'amt': round(float(row['amt']), 2),
                'category': row['category'],
                'age': int(row['age']),
                'hour': int(row['hour']),
                'score': score,
                'label': label
            })
        except:
            continue

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)