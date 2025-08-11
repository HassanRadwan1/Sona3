# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os

COLUMN_RENAME_MAP = {
    "مجموع_المعاش": "إجمالي المعاش",
    "إجمالي دخل الأسرة": "الدخل الكلي للأسرة",
    "مجموع_الغاز": "غاز / أنابيب (شهريًا)",
    "مجموع_الأكل": "متوسط الأكل (شهريًا)",
}

LABEL_RENAME_MAP = {
    "تقييم الكراتين": "احتياج كراتين",
    "تقييم الكسوة": "احتياج كسوة",
    'تقييم المشروع ': "احتياج مشروع",
    'تقييم الشهرية ': "احتياج شهرية",
    # زوّد أو عدّل الأسماء اللي تحبها هنا
}

MODEL_DIR = os.path.dirname(__file__)
TARGET_COL = "الأسرة"

@st.cache_resource
def load_models(p):
    clf1 = joblib.load(os.path.join(p, 'accept_model.pkl'))
    label_models = joblib.load(os.path.join(p, 'labels_models.pkl'))
    if "تقييم عدد الألحفة" in label_models:
        label_models.pop("تقييم عدد الألحفة")
    return clf1, label_models

clf1, label_models = load_models(MODEL_DIR)
meta1 = json.load(open(os.path.join(MODEL_DIR, 'meta_stage1.json'), 'r', encoding='utf-8'))
meta2 = json.load(open(os.path.join(MODEL_DIR, 'meta_stage2.json'), 'r', encoding='utf-8'))

feat1 = meta1['feature_order']
feat2 = meta2['feature_order']

EXCLUDE_COLS = ["تقييم السقف", "تقييم وصلة المياه", "تقييم عدد الألحفة"]
EVALUATION_COLS = [c for c in label_models.keys() if c not in EXCLUDE_COLS]

def explain_rejection(row: pd.Series) -> list[str]:
    income_cols = [
        'إجمالي دخل رب الأسرة (شهريًا)',
        'إجمالي دخل الزوج/ة  (شهريًا)',
        'إجمالي دخل باقي أفراد الأسرة  (شهريًا)',
        'تكافل وكرامة ',
        'مجموع_المعاش',
        'إجمالي دخل الأسرة'
    ]
    expense_cols = [
        'كهربا (شهريًا)',
        'مياه (شهريًا)',
        'مجموع_الغاز',
        'مجموع_الأكل',
        'إيجار البيت ( لو البيت إيجار )',
        'قيمة قسط أو جمعية في الشهر',
        'إجمالي مصاريف التعليم (شهريًا)',
        'إجمالي مصاريف العلاج (شهريًا)',
        'إجمالي مصاريف الأسرة'
    ]
    def getv(col):
        return float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0
    total_income = getv('إجمالي دخل الأسرة')
    if total_income <= 0:
        total_income = sum(getv(c) for c in income_cols if c != 'إجمالي دخل الأسرة')
    total_expenses = getv('إجمالي مصاريف الأسرة')
    if total_expenses <= 0:
        total_expenses = sum(getv(c) for c in expense_cols if c != 'إجمالي مصاريف الأسرة')
    rent = getv('إيجار البيت ( لو البيت إيجار )')
    edu = getv('إجمالي مصاريف التعليم (شهريًا)')
    med = getv('إجمالي مصاريف العلاج (شهريًا)')
    food = getv('مجموع_الأكل')
    elec = getv('كهربا (شهريًا)')
    water = getv('مياه (شهريًا)')
    gas = getv('مجموع_الغاز')
    bills = elec + water + gas
    installment = getv('قيمة قسط أو جمعية في الشهر')
    reasons = []
    inc = max(total_income, 1.0)
    if total_expenses > total_income:
        reasons.append("إجمالي المصاريف أعلى من إجمالي الدخل")
    if rent > 0 and rent >= 0.30 * inc:
        reasons.append(f"الإيجار مرتفع ({rent:.0f}) ويمثل ≥30% من الدخل")
    if food >= 0.40 * inc:
        reasons.append(f"مصاريف الأكل مرتفعة ({food:.0f}) وتمثل ≥40% من الدخل")
    if bills >= 0.15 * inc:
        reasons.append(f"فواتير الكهرباء/المياه/الغاز مرتفعة ({bills:.0f}) وتمثل ≥15% من الدخل")
    if (edu + med) >= 0.20 * inc:
        reasons.append(f"التعليم/العلاج مرتفعان ({edu+med:.0f}) ويمثلان ≥20% من الدخل")
    if installment >= 0.20 * inc:
        reasons.append(f"القسط/الجمعية مرتفعة ({installment:.0f}) وتمثل ≥20% من الدخل")
    if total_income <= 0:
        reasons.append("الدخل الكلي غير مُحدد أو مساوي للصفر")
    if not reasons:
        reasons.append("احتمال القبول أقل من العتبة المطلوبة وفقًا لمدخلات الدخل/المصاريف")
    return reasons

def predict_df(df: pd.DataFrame, threshold: float = 0.80, margin: float = 0.05, prioritize_reject: bool = True):
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    X1 = df.reindex(columns=feat1, fill_value=0)
    classes = np.array(clf1.classes_)
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = int(np.argmax(classes))
    proba_pos = clf1.predict_proba(X1)[:, pos_idx]
    proba_neg = 1.0 - proba_pos if set(classes.tolist()) == {0,1} else None
    cut = threshold + (margin if prioritize_reject else 0.0)
    acc_mask = (proba_pos >= cut).astype(int)
    preds = {TARGET_COL: acc_mask, 'proba_accepted': proba_pos}
    if proba_neg is not None:
        preds['proba_rejected'] = proba_neg
    for c in EVALUATION_COLS:
        preds[c] = np.zeros(len(df), dtype=int)
    idx = np.where(acc_mask == 1)[0]
    if len(idx) > 0:
        X2 = df.reindex(columns=feat2, fill_value=0).iloc[idx]
        for c, clf in label_models.items():
            if c in EVALUATION_COLS:
                preds[c][idx] = clf.predict(X2)
    return pd.DataFrame(preds), float(cut)






st.markdown(
    "<h1 style='text-align: center;'>اختبار القبول – صناع الحياة</h1>",
    unsafe_allow_html=True
)

input_data = {}
for col in feat1:
    display_name = COLUMN_RENAME_MAP.get(col, col)
    input_data[col] = st.number_input(display_name, value=0.0, step=1.0)

if st.button("✅ Predict"):
    row_df = pd.DataFrame([input_data])
    if (row_df == 0).all(axis=1).iloc[0]:
        st.error("⚠️ يجب إدخال بيانات على الأقل في حقل واحد قبل التنبؤ.")
    else:
        preds, _ = predict_df(row_df)
        decision = int(preds.loc[0, TARGET_COL])
        if decision == 0:
            st.error("❌ مرفوضة")
            reasons = explain_rejection(row_df.iloc[0])
            st.write("**أسباب الرفض المحتملة:**")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.success("✅ مقبولة")

            def _norm(s: str) -> str:
                s = str(s)
                rep = {"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه"}
                for k,v in rep.items(): s = s.replace(k,v)
                return " ".join(s.split())

            excluded_norms = {_norm("تقييم عدد الألحفة"), _norm("تقييم عدد الالحفه")}
            eval_cols = []
            for c in preds.columns:
                if c == TARGET_COL:
                    continue
                if c.startswith("تقييم"):
                    if _norm(c) in excluded_norms:
                        continue
                    eval_cols.append(c)

            bad_cols = [c for c in preds.columns if _norm(c) in excluded_norms]
            if bad_cols:
                preds = preds.drop(columns=bad_cols)

            on_labels = [c for c in eval_cols if int(preds.loc[0, c]) == 1]

            st.write("**الأسرة محتاجة:**", "، ".join([LABEL_RENAME_MAP.get(c, c) for c in on_labels]) if on_labels else "لا يوجد")
            if eval_cols and any(int(preds.loc[0, c]) == 1 for c in eval_cols):
                st.dataframe(preds[eval_cols].rename(columns=LABEL_RENAME_MAP), use_container_width=True)
