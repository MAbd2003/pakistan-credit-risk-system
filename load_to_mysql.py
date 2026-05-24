# ============================================================
# CREDIT RISK — MYSQL DATA PIPELINE
# Author: Muhammad Abdullah
# Purpose: Load cleaned loan data into MySQL database
# ============================================================

import pandas as pd
import numpy as np
import pymysql
import warnings
warnings.filterwarnings('ignore')

# ── CONNECT TO MYSQL ──────────────────────────────────────
print("Connecting to MySQL...")
try:
    conn = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password="an#654321",
        database='credit_risk_db',
        connect_timeout=5
    )
    cursor = conn.cursor()
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

# ── LOAD AND CLEAN DATA ───────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv('credit_risk_dataset.csv')

df = df[df['person_age'] <= 100]
df = df[df['person_emp_length'] <= 60]
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df = df.dropna()

print(f"Records ready to load: {len(df):,}")

# ── ADD RISK LEVEL COLUMN ─────────────────────────────────
def assign_risk(row):
    score = 0
    if row['loan_percent_income'] > 0.3: score += 3
    elif row['loan_percent_income'] > 0.15: score += 1
    if row['loan_grade'] in ['E', 'F', 'G']: score += 3
    elif row['loan_grade'] in ['C', 'D']: score += 1
    if row['cb_person_default_on_file'] == 'Y': score += 2
    if row['loan_int_rate'] > 15: score += 2
    if row['person_emp_length'] < 2: score += 1
    if score >= 6: return 'HIGH RISK'
    elif score >= 3: return 'MEDIUM RISK'
    else: return 'LOW RISK'

df['risk_level'] = df.apply(assign_risk, axis=1)

print(f"\nRisk Distribution:")
print(df['risk_level'].value_counts())

# ── LOAD INTO MYSQL ───────────────────────────────────────
print("\nLoading data into MySQL...")

insert_query = """
INSERT INTO loans (
    person_age, person_income, person_home_ownership,
    person_emp_length, loan_intent, loan_grade,
    loan_amnt, loan_int_rate, loan_status,
    loan_percent_income, cb_person_default_on_file,
    cb_person_cred_hist_length, risk_level
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

batch = []
count = 0

for _, row in df.iterrows():
    batch.append((
        int(row['person_age']),
        int(row['person_income']),
        str(row['person_home_ownership']),
        float(row['person_emp_length']),
        str(row['loan_intent']),
        str(row['loan_grade']),
        int(row['loan_amnt']),
        float(row['loan_int_rate']),
        int(row['loan_status']),
        float(row['loan_percent_income']),
        str(row['cb_person_default_on_file']),
        int(row['cb_person_cred_hist_length']),
        str(row['risk_level'])
    ))
    count += 1
    if count % 1000 == 0:
        cursor.executemany(insert_query, batch)
        conn.commit()
        batch = []
        print(f"  Loaded {count:,} records...")

if batch:
    cursor.executemany(insert_query, batch)
    conn.commit()

print(f"\nTotal records loaded: {count:,}")

# ── VERIFY WITH SQL QUERIES ───────────────────────────────
print("\n" + "="*50)
print("VERIFICATION QUERIES")
print("="*50)

cursor.execute("SELECT COUNT(*) FROM loans")
print(f"\nTotal records in MySQL: {cursor.fetchone()[0]:,}")

cursor.execute("""
    SELECT risk_level, COUNT(*) as count,
    ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM loans), 2) as percentage
    FROM loans GROUP BY risk_level ORDER BY count DESC
""")
print("\nRisk Distribution in MySQL:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,} records ({row[2]}%)")

cursor.execute("""
    SELECT loan_grade,
    ROUND(AVG(loan_int_rate), 2) as avg_rate,
    ROUND(AVG(loan_percent_income)*100, 2) as avg_income_pct,
    SUM(loan_status) as total_defaults
    FROM loans GROUP BY loan_grade ORDER BY loan_grade
""")
print("\nLoan Grade Analysis:")
for row in cursor.fetchall():
    print(f"  Grade {row[0]}: Avg Rate {row[1]}% | Income% {row[2]}% | Defaults {row[3]:,}")

cursor.close()
conn.close()
print("\nMySQL pipeline complete!")
print("Database ready for Power BI connection!") 