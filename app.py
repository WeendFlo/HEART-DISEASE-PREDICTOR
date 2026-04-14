"""
Heart Disease Risk Prediction — Flask Web Application
======================================================
Features:
  - User authentication (register/login)
  - Prediction form with all 3 ML models
  - Explainable predictions (feature importance)
  - Health dashboard (stats, charts, model comparison)
  - Prediction history per user
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'heart-disease-prediction-secret-key-2025'

# ============================================================
# LOAD ML MODELS
# ============================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    with open(os.path.join(MODEL_DIR, filename), 'rb') as f:
        return pickle.load(f)

models = {
    'logistic_regression': load_pickle('logistic_regression.pkl'),
    'decision_tree': load_pickle('decision_tree.pkl'),
    'random_forest': load_pickle('random_forest.pkl'),
}

scaler = load_pickle('scaler.pkl')
feature_names = load_pickle('features.pkl')
model_performance = load_pickle('model_performance.pkl')

# ============================================================
# SIMPLE USER & PREDICTION STORAGE (JSON file-based)
# ============================================================
USERS_FILE = os.path.join(MODEL_DIR, 'users.json')
PREDICTIONS_FILE = os.path.join(MODEL_DIR, 'predictions.json')

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# ============================================================
# AUTH DECORATOR
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ============================================================
# FEATURE INFO (for the form and explanations)
# ============================================================
FEATURE_INFO = {
    'age':      {'label': 'Your Age', 'unit': 'years', 'min': 20, 'max': 90, 'step': 1, 'default': 50,
                 'help': 'How old are you?'},
    'sex':      {'label': 'Gender', 'type': 'select', 'options': {'1': 'Male', '0': 'Female'},
                 'help': 'Select your biological sex'},
    'cp':       {'label': 'Do You Experience Chest Pain?', 'type': 'select',
                 'options': {
                     '0': 'No chest pain at all',
                     '1': 'Sometimes mild chest discomfort',
                     '2': 'Chest pain not related to heart',
                     '3': 'Yes, squeezing/pressure pain in chest'
                 },
                 'help': 'Describe your chest pain experience. Choose "No chest pain" if none.'},
    'trestbps': {'label': 'Blood Pressure', 'unit': 'mm Hg', 'min': 80, 'max': 220, 'step': 1, 'default': 130,
                 'help': 'Your resting BP. Normal: ~120. Find it on your last checkup report.'},
    'chol':     {'label': 'Cholesterol Level', 'unit': 'mg/dl', 'min': 100, 'max': 600, 'step': 1, 'default': 230,
                 'help': 'Total cholesterol from blood test. Normal: below 200.'},
    'fbs':      {'label': 'Is Your Blood Sugar High?', 'type': 'select',
                 'options': {'0': 'No (normal blood sugar)', '1': 'Yes (diabetic / high sugar)'},
                 'help': 'Do you have diabetes or high fasting blood sugar (>120 mg/dl)?'},
    'restecg':  {'label': 'Heart Test (ECG) Result', 'type': 'select',
                 'options': {
                     '0': 'Normal / Never had an ECG',
                     '1': 'Some irregularity detected',
                     '2': 'Heart enlargement detected'
                 },
                 'help': 'Result from an ECG heart test. Pick "Normal" if you have not had one.'},
    'thalch':   {'label': 'Highest Heart Rate During Exercise', 'unit': 'bpm', 'min': 60, 'max': 220, 'step': 1, 'default': 150,
                 'help': 'Your peak heart rate when active. Estimate: 220 minus your age.'},
    'exang':    {'label': 'Chest Pain When Exercising?', 'type': 'select',
                 'options': {'0': 'No, I feel fine during exercise', '1': 'Yes, I get chest pain when active'},
                 'help': 'Do you feel chest pain, pressure or tightness during physical activity?'},
    'oldpeak':  {'label': 'Heart Stress Test Score', 'unit': '', 'min': 0, 'max': 7, 'step': 0.1, 'default': 1.0,
                 'help': 'From a treadmill/stress test report. Enter 0 if you have never had one.'},
}

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('predict'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('register.html')

        users = load_json(USERS_FILE)
        if username in users:
            flash('Username already exists. Try a different one.', 'error')
            return render_template('register.html')

        users[username] = {
            'password': password,  # In production, hash this!
            'name': name or username,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        save_json(USERS_FILE, users)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        users = load_json(USERS_FILE)
        if username in users and users[username]['password'] == password:
            session['user'] = username
            session['name'] = users[username]['name']
            flash(f'Welcome back, {users[username]["name"]}!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None

    if request.method == 'POST':
        try:
            # Collect form data
            patient_data = {}
            for feat in feature_names:
                val = request.form.get(feat)
                patient_data[feat] = float(val)

            selected_model = request.form.get('model', 'decision_tree')

            # Create DataFrame and scale
            input_df = pd.DataFrame([patient_data])
            input_scaled = scaler.transform(input_df)

            # Get predictions from ALL models
            all_predictions = {}
            for name, model in models.items():
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0]
                all_predictions[name] = {
                    'prediction': int(pred),
                    'probability': float(prob[1]),
                    'label': 'High Risk' if pred == 1 else 'Low Risk'
                }

            # Primary result from selected model
            primary = all_predictions[selected_model]

            # Feature importance / explanation
            explanation = get_explanation(patient_data, input_scaled, selected_model)

            result = {
                'primary': primary,
                'selected_model': selected_model,
                'all_predictions': all_predictions,
                'patient_data': patient_data,
                'explanation': explanation
            }

            # Save prediction to history
            save_prediction(session['user'], patient_data, all_predictions, selected_model)

        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')

    return render_template('predict.html',
                           features=FEATURE_INFO,
                           feature_names=feature_names,
                           result=result,
                           model_perf=model_performance)


def get_explanation(patient_data, input_scaled, model_name):
    """Generate human-readable explanation of which features influenced the prediction most."""
    explanations = []

    # Use feature importances for tree-based models
    model = models[model_name]

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return []

    # Pair features with their importance
    feat_imp = list(zip(feature_names, importances, input_scaled[0]))
    feat_imp.sort(key=lambda x: x[1], reverse=True)

    # Generate explanations for top features
    thresholds = {
        'age': (55, 'above', 'Patient age is above 55, increasing risk'),
        'trestbps': (140, 'above', 'Blood pressure is elevated (>140 mm Hg)'),
        'chol': (240, 'above', 'Cholesterol is high (>240 mg/dl)'),
        'thalch': (140, 'below', 'Max heart rate is below normal (<140 bpm)'),
        'oldpeak': (1.5, 'above', 'Significant ST depression during exercise'),
        'cp': (2.5, 'above', 'Asymptomatic chest pain type is concerning'),
        'exang': (0.5, 'above', 'Exercise-induced angina is present'),
        'fbs': (0.5, 'above', 'Elevated fasting blood sugar'),
    }

    for feat, importance, scaled_val in feat_imp[:5]:
        pct = importance / sum(importances) * 100
        value = patient_data[feat]
        label = FEATURE_INFO[feat]['label']

        # Determine risk direction
        risk_level = 'neutral'
        detail = f'{label}: {value}'

        if feat in thresholds:
            threshold, direction, message = thresholds[feat]
            if direction == 'above' and value > threshold:
                risk_level = 'high'
                detail = message
            elif direction == 'below' and value < threshold:
                risk_level = 'high'
                detail = message
            elif direction == 'above' and value <= threshold:
                risk_level = 'low'
                detail = f'{label} is within normal range'
            elif direction == 'below' and value >= threshold:
                risk_level = 'low'
                detail = f'{label} is within normal range'

        explanations.append({
            'feature': label,
            'importance': round(pct, 1),
            'risk_level': risk_level,
            'detail': detail,
            'value': value
        })

    return explanations


def save_prediction(username, patient_data, predictions, selected_model):
    """Save prediction to history."""
    all_preds = load_json(PREDICTIONS_FILE)
    if username not in all_preds:
        all_preds[username] = []

    all_preds[username].append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_data': patient_data,
        'predictions': {k: {'prediction': v['prediction'], 'probability': v['probability']}
                       for k, v in predictions.items()},
        'selected_model': selected_model,
        'result': predictions[selected_model]['label']
    })

    save_json(PREDICTIONS_FILE, all_preds)


@app.route('/dashboard')
@login_required
def dashboard():
    all_preds = load_json(PREDICTIONS_FILE)
    user_preds = all_preds.get(session['user'], [])

    # Calculate stats
    total = len(user_preds)
    high_risk = sum(1 for p in user_preds if p['result'] == 'High Risk')
    low_risk = total - high_risk

    # All predictions across all users (for global stats)
    all_user_preds = []
    for preds in all_preds.values():
        all_user_preds.extend(preds)

    global_total = len(all_user_preds)
    global_high = sum(1 for p in all_user_preds if p['result'] == 'High Risk')
    global_low = global_total - global_high

    # Feature importance for ALL models
    all_feat_importance = {}
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            continue
        feat_imp = list(zip(feature_names, importances))
        feat_imp.sort(key=lambda x: x[1], reverse=True)
        all_feat_importance[model_name] = [
            {'name': FEATURE_INFO[f]['label'], 'value': round(v / sum(importances) * 100, 1)}
            for f, v in feat_imp
        ]

    stats = {
        'user_total': total,
        'user_high': high_risk,
        'user_low': low_risk,
        'global_total': global_total,
        'global_high': global_high,
        'global_low': global_low,
        'model_performance': model_performance,
        'all_feat_importance': all_feat_importance,
        'recent_predictions': user_preds[-10:][::-1]
    }

    return render_template('dashboard.html', stats=stats)


@app.route('/history')
@login_required
def history():
    all_preds = load_json(PREDICTIONS_FILE)
    user_preds = all_preds.get(session['user'], [])
    return render_template('history.html', predictions=user_preds[::-1])


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
