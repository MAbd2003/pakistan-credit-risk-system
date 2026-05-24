# ============================================================
# PAKISTAN CREDIT RISK — FLASK WEB APPLICATION
# Author: Muhammad Abdullah
# Purpose: REST API + Web Interface for Credit Risk Scoring
# ============================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── TRAIN MODEL ON STARTUP ────────────────────────────────
print("Loading and training model...")

df = pd.read_csv('credit_risk_dataset.csv')
df = df[df['person_age'] <= 100]
df = df[df['person_emp_length'] <= 60]
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df = df.dropna()

df_ml = df.copy()
le = LabelEncoder()
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col])

X = df_ml.drop('loan_status', axis=1)
y = df_ml['loan_status']

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, random_state=42,
                      eval_metric='logloss', verbosity=0)
model.fit(X_train, y_train)
print("Model ready!")

# ── ENCODING MAPS ─────────────────────────────────────────
home_map = {'RENT': 3, 'MORTGAGE': 0, 'OWN': 2, 'OTHER': 1}
intent_map = {'PERSONAL': 4, 'EDUCATION': 1, 'MEDICAL': 3,
              'VENTURE': 5, 'HOMEIMPROVEMENT': 2, 'DEBTCONSOLIDATION': 0}
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_map = {'N': 0, 'Y': 1}

# ── ROUTES ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        age = int(data['age'])
        income = int(data['income'])
        home = data['home_ownership']
        emp = float(data['emp_length'])
        intent = data['intent']
        grade = data['grade']
        amount = int(data['loan_amount'])
        rate = float(data['int_rate'])
        default_file = data['default_on_file']
        cred_hist = int(data['cred_hist'])
        loan_pct = amount / income

        features = np.array([[
            age, income,
            home_map.get(home, 3),
            emp,
            intent_map.get(intent, 4),
            grade_map.get(grade, 2),
            amount, rate, loan_pct,
            default_map.get(default_file, 0),
            cred_hist
        ]])

        features_scaled = scaler.transform(features)
        probability = float(model.predict_proba(features_scaled)[0][1]) * 100
        prediction = int(model.predict(features_scaled)[0])

        if probability < 20:
            risk = "LOW RISK"
            decision = "APPROVE LOAN"
            color = "success"
            icon = "✅"
        elif probability < 50:
            risk = "MEDIUM RISK"
            decision = "CONDITIONAL APPROVAL"
            color = "warning"
            icon = "⚠️"
        else:
            risk = "HIGH RISK"
            decision = "REJECT LOAN"
            color = "danger"
            icon = "❌"

        return jsonify({
            'probability': round(probability, 1),
            'risk': risk,
            'decision': decision,
            'color': color,
            'icon': icon,
            'prediction': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 