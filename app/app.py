from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='.')   # index.html lives next to app.py
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD HELPER
# ══════════════════════════════════════════════════════════════════════════════
def load(path):
    if os.path.exists(path):
        return joblib.load(path)
    print(f"[WARN] Not found: {path}")
    return None

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
BASE_REG = r"C:\Users\LENOVO\Desktop\project_ml_retail\models\linear"

reg_imputer_median = load(os.path.join(BASE_REG, "operation", "imputer_median.pkl"))
reg_scaler_knn     = load(os.path.join(BASE_REG, "operation", "scaler_knn.pkl"))
reg_imputer_knn    = load(os.path.join(BASE_REG, "operation", "imputer_knn.pkl"))
reg_encoder_ord    = load(os.path.join(BASE_REG, "operation", "encoder_ordinal.pkl"))
reg_encoder_ohe    = load(os.path.join(BASE_REG, "operation", "encoder_ohe.pkl"))
reg_encoder_target = load(os.path.join(BASE_REG, "operation", "encoder_target.pkl"))
reg_scaler_final   = load(os.path.join(BASE_REG, "operation", "scaler_final.pkl"))
reg_model          = load(os.path.join(BASE_REG, "regression", "best_model.pkl"))

# ══════════════════════════════════════════════════════════════════════════════
# CHURN ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
BASE_CHURN = r"C:\Users\LENOVO\Desktop\project_ml_retail\models\tree"

churn_imputer_median = load(os.path.join(BASE_CHURN, "operation", "imputer_median.pkl"))
churn_scaler_knn     = load(os.path.join(BASE_CHURN, "operation", "scaler_knn.pkl"))
churn_imputer_knn    = load(os.path.join(BASE_CHURN, "operation", "imputer_knn.pkl"))
churn_imputer_geo    = load(os.path.join(BASE_CHURN, "operation", "imputer_geo.pkl"))
churn_encoder_ord    = load(os.path.join(BASE_CHURN, "operation", "encoder_ordinal.pkl"))
churn_encoder_ohe    = load(os.path.join(BASE_CHURN, "operation", "encoder_ohe.pkl"))
churn_encoder_target = load(os.path.join(BASE_CHURN, "operation", "encoder_target.pkl"))
churn_model          = load(os.path.join(BASE_CHURN, "classification", "xgb_model.pkl"))

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Regression ────────────────────────────────────────────────────────────────
REG_COLS_MEDIAN  = ['AvgDaysBetweenPurchases', 'SupportTicketsCount',
                    'SatisfactionScore', 'SupportBurden']
REG_COLS_KNN     = ['Age', 'Frequency', 'CustomerTenureDays']
REG_ORDINAL_COLS = ['LoyaltyLevel', 'BasketSizeCategory']
REG_OHE_COLS     = ['Gender', 'ProductDiversity', 'Region']
REG_TARGET_COLS  = ['Country']

REG_FEATURE_COLS = [
    'Frequency', 'AvgQuantityPerTransaction', 'CustomerTenureDays',
    'FirstPurchaseDaysAgo', 'PreferredDayOfWeek', 'PreferredMonth',
    'WeekendPurchaseRatio', 'AvgDaysBetweenPurchases', 'UniqueProducts',
    'AvgProductsPerTransaction', 'ZeroPriceCount', 'CancelledTransactions',
    'ReturnRatio', 'Age', 'SupportTicketsCount', 'SatisfactionScore',
    'LoyaltyLevel', 'BasketSizeCategory', 'Country', 'ZeroPriceRatio',
    'SupportBurden', 'Gender_F', 'Gender_M', 'Gender_Unknown',
    'ProductDiversity_Explorateur', 'ProductDiversity_Modéré',
    'ProductDiversity_Spécialisé', 'Region_Afrique',
    'Region_Amérique du Nord', 'Region_Asie', 'Region_Autre',
    'Region_Europe centrale', 'Region_Europe continentale',
    "Region_Europe de l'Est", 'Region_Europe du Nord',
    'Region_Europe du Sud', 'Region_Moyen-Orient', 'Region_Océanie',
    'Region_UK'
]

# ── Churn ─────────────────────────────────────────────────────────────────────
CHURN_COLS_MEDIAN  = ['AvgDaysBetweenPurchases', 'SupportTicketsCount', 'SatisfactionScore']
CHURN_COLS_KNN     = ['Age', 'Frequency', 'MonetaryTotal']
CHURN_COLS_GEO     = ['GeoIP']
CHURN_ORDINAL_COLS = ['AgeCategory', 'SpendingCategory', 'LoyaltyLevel',
                      'BasketSizeCategory', 'PreferredTimeOfDay']
CHURN_OHE_COLS     = ['FavoriteSeason', 'WeekendPreference', 'ProductDiversity', 'Region']
CHURN_TARGET_COLS  = ['Country', 'GeoIP']
CHURN_LEAKY        = ['Recency', 'TenureRatio', 'RFMSegment']

XGB_FEATURE_COLS = [
    'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'MonetaryStd', 'MonetaryMin',
    'MonetaryMax', 'TotalQuantity', 'AvgQuantityPerTransaction', 'MinQuantity',
    'MaxQuantity', 'FirstPurchaseDaysAgo', 'PreferredDayOfWeek', 'PreferredHour',
    'WeekendPurchaseRatio', 'AvgDaysBetweenPurchases', 'UniqueDescriptions',
    'AvgProductsPerTransaction', 'NegativeQuantityCount', 'ReturnRatio',
    'TotalTransactions', 'UniqueInvoices', 'AvgLinesPerInvoice',
    'SupportTicketsCount', 'SatisfactionScore', 'AgeCategory', 'SpendingCategory',
    'RegYear', 'RegMonth', 'GeoIP', 'MonetaryPerDay', 'AvgBasketValue',
    'CancelRate', 'ProductsPerTrans', 'ProductDiversity_Explorateur'
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def assign_age_category(age):
    if pd.isna(age): return 'Unknown'
    age = int(age)
    if age < 25:  return '18-24'
    if age < 35:  return '25-34'
    if age < 45:  return '35-44'
    if age < 55:  return '45-54'
    if age < 65:  return '55-64'
    return '65+'

def churn_response(pred, proba):
    risk_pct = round(float(proba) * 100, 1)
    if risk_pct >= 70:
        level, color, action = "High Risk",   "#e74c3c", "Immediate retention campaign needed"
    elif risk_pct >= 40:
        level, color, action = "Medium Risk", "#e67e22", "Monitor — send re-engagement offer"
    else:
        level, color, action = "Low Risk",    "#2ecc71", "Maintain loyalty programme"
    return {"churn": int(pred), "probability": risk_pct,
            "risk_level": level, "color": color, "action": action}

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


# ── REGRESSION ────────────────────────────────────────────────────────────────
@app.route('/api/predict/regression', methods=['POST'])
def predict_regression():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400
        df = pd.DataFrame([data])

        # Step 1: Feature engineering
        df['ZeroPriceRatio'] = df['ZeroPriceCount'] / (df['TotalTransactions'] + 1)
        df['SupportBurden']  = df['SupportTicketsCount'] / (df['CustomerTenureDays'] + 1)

        # Step 2: Median imputation
        if reg_imputer_median:
            df[REG_COLS_MEDIAN] = reg_imputer_median.transform(df[REG_COLS_MEDIAN])

        # Step 3: KNN imputation
        if reg_scaler_knn and reg_imputer_knn:
            scaled  = reg_scaler_knn.transform(df[REG_COLS_KNN])
            imputed = reg_imputer_knn.transform(scaled)
            df[REG_COLS_KNN] = reg_scaler_knn.inverse_transform(imputed)

        # Step 4: Ordinal encoding
        if reg_encoder_ord:
            df[REG_ORDINAL_COLS] = reg_encoder_ord.transform(df[REG_ORDINAL_COLS])

        # Step 5: OHE
        if reg_encoder_ohe:
            ohe_arr   = reg_encoder_ohe.transform(df[REG_OHE_COLS])
            ohe_names = reg_encoder_ohe.get_feature_names_out(REG_OHE_COLS)
            ohe_df    = pd.DataFrame(ohe_arr, columns=ohe_names, index=df.index)
            df = pd.concat([df.drop(columns=REG_OHE_COLS), ohe_df], axis=1)

        # Step 6: Target encoding
        if reg_encoder_target:
            df[REG_TARGET_COLS] = reg_encoder_target.transform(df[REG_TARGET_COLS])

        # Step 7: Align columns
        df = df.reindex(columns=REG_FEATURE_COLS, fill_value=0)

        # Step 8: Scale
        X = reg_scaler_final.transform(df) if reg_scaler_final else df.values

        # Step 9: Predict
        prediction = float(reg_model.predict(X)[0])

        if prediction >= 60:   tier, color = "High Value",   "#2ecc71"
        elif prediction >= 30: tier, color = "Medium Value", "#e67e22"
        else:                  tier, color = "Low Value",    "#e74c3c"

        return jsonify({
            "prediction": round(prediction, 2),
            "tier":       tier,
            "color":      color,
            "message":    f"Predicted average spend: £{prediction:.2f} per transaction"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── CHURN ─────────────────────────────────────────────────────────────────────
@app.route('/api/predict/churn', methods=['POST'])
def predict_churn():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400
        df = pd.DataFrame([data])

        # Step 1: Feature engineering — only compute when raw inputs are present.
        # The form may send pre-computed values directly; never overwrite them
        # with a computation that requires columns not in the form.
        if 'CancelledTransactions' in df.columns and 'TotalTransactions' in df.columns:
            df['CancelRate'] = df['CancelledTransactions'] / (df['TotalTransactions'] + 1)
        if 'UniqueProducts' in df.columns and 'TotalTransactions' in df.columns:
            df['ProductsPerTrans'] = df['UniqueProducts'] / (df['TotalTransactions'] + 1)
        if 'CustomerTenureDays' in df.columns and 'MonetaryTotal' in df.columns:
            df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['CustomerTenureDays'] + 1)
        if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
            df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)

        # Step 2: Median imputation for the 3 skewed numeric cols (all present in form)
        if churn_imputer_median:
            df[CHURN_COLS_MEDIAN] = churn_imputer_median.transform(df[CHURN_COLS_MEDIAN])

        # Steps 3-9 (KNN imputation, AgeCategory, GeoIP, CountryMismatch,
        # ordinal encoding, OHE, target encoding) are skipped here because:
        #  - Age, GeoIP, Country, FavoriteSeason, WeekendPreference, Region etc.
        #    are not sent by the simplified churn form.
        #  - All Region/FavoriteSeason/WeekendPreference OHE columns and Country
        #    were dropped as zero-importance before the final model fit (notebook
        #    cell 43), so encoding them would produce columns the model ignores.
        #  - reindex(fill_value=0) below handles any remaining missing features.

        # Align to the exact 34 features XGBoost was trained on
        df = df.reindex(columns=XGB_FEATURE_COLS, fill_value=0)

        # Cast ordinal columns to int so XGBoost sees numeric dtype
        for col in ['SpendingCategory', 'AgeCategory', 'GeoIP']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Step 12: Predict
        pred  = churn_model.predict(df)[0]
        proba = churn_model.predict_proba(df)[0][1]
        return jsonify(churn_response(pred, proba))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)