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

        # ── SBP REGULATION CHECKS ─────────────────────────
        sbp_flags = []
        sbp_reject = False

        # SBP Rule 1: Debt Burden Ratio (DBR)
        # Monthly installment cannot exceed 50% of monthly income
        monthly_income = income / 12
        monthly_installment = (amount * (rate/100/12)) / (1 - (1 + rate/100/12)**-36)
        dbr = (monthly_installment / monthly_income) * 100

        if dbr > 50:
            sbp_reject = True
            sbp_flags.append(f"DBR VIOLATION: Monthly installment ({dbr:.1f}% of income) exceeds SBP limit of 50%")
        else:
            sbp_flags.append(f"DBR CHECK PASSED: {dbr:.1f}% of monthly income (SBP limit: 50%)")

        # SBP Rule 2: Clean Lending Limit
        # Unsecured loans cannot exceed PKR 500,000
        if amount > 500000 and home == 'RENT':
            sbp_flags.append(f"CLEAN LENDING FLAG: Loan amount exceeds SBP Rs.500,000 clean lending limit. Collateral required.")

        # SBP Rule 3: Age eligibility
        if age < 18 or age > 65:
            sbp_reject = True
            sbp_flags.append(f"AGE VIOLATION: Applicant age {age} outside SBP eligible range of 18-65 years")
        else:
            sbp_flags.append(f"AGE CHECK PASSED: {age} years (SBP eligible range: 18-65)")

        # SBP Rule 4: Previous default check (eCIB simulation)
        if default_file == 'Y':
            sbp_flags.append("eCIB FLAG: Previous default on record. SBP requires banks to review eCIB report before approval.")

        # SBP Rule 5: Employment stability
        if emp < 1:
            sbp_flags.append("EMPLOYMENT FLAG: Less than 1 year employment. SBP recommends minimum 1 year stable income.")

        # ── ML PREDICTION ─────────────────────────────────
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

        # ── FINAL DECISION ────────────────────────────────
        if sbp_reject:
            risk = "SBP REGULATORY REJECT"
            decision = "REJECT — SBP VIOLATION"
            color = "danger"
            icon = "🚫"
        elif probability < 20:
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
            'prediction': prediction,
            'sbp_flags': sbp_flags,
            'dbr': round(dbr, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 
