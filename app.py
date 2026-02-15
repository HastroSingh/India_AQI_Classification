import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="India AQI Classification", page_icon="~", layout="wide")

MODEL_DIR = "model"

@st.cache_resource
def load_models():
    files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'KNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest (Ensemble)': 'random_forest.pkl',
        'XGBoost (Ensemble)': 'xgboost_model.pkl'
    }
    loaded = {}
    for name, fname in files.items():
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            loaded[name] = joblib.load(p)
    return loaded

@st.cache_resource
def load_scaler():
    p = os.path.join(MODEL_DIR, 'scaler.pkl')
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def load_le():
    p = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data
def load_metrics():
    p = os.path.join(MODEL_DIR, 'metrics.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

@st.cache_data
def load_sample():
    p = os.path.join(MODEL_DIR, 'test_sample.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

models = load_models()
scaler = load_scaler()
le = load_le()
metrics_data = load_metrics()
sample_data = load_sample()

# sidebar
st.sidebar.title("India AQI Classifier")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Home", "Model Comparison", "Predictions"])
st.sidebar.markdown("---")
st.sidebar.info("ML Assignment 2\n\n**Dataset:** India Air Quality (2015-2020)\n\n**Models:** 6 classifiers\n\n**Name:** Harsh Deep Singh\n\n**Roll No:** 2025ab05204\n\n**Email:** 2025ab05204@wilp.bits-pilani.ac.in\n\n**BITS Pilani**")

# ---- HOME ----
if page == "Home":
    st.title("India Air Quality Index (AQI) Classification")
    st.caption("ML Classification - Assignment 2")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", "12")
    inst = f"{metrics_data['dataset_info']['instances']:,}" if metrics_data else "24,000+"
    c2.metric("Instances", inst)
    c3.metric("Models", "6")
    c4.metric("Type", "Multi-class")

    st.markdown("---")
    st.header("Problem Statement")
    st.write(
        "Air pollution is a major public health issue in India. This project classifies "
        "air quality into 6 AQI categories (Good to Severe) using pollutant concentration "
        "data from monitoring stations across India. Six ML models are compared."
    )

    st.header("Dataset")
    st.write(
        "- **Source:** [Kaggle - Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)\n"
        "- **Features:** 12 pollutant measurements (PM2.5, PM10, NO, NO2, etc.)\n"
        "- **Target:** AQI_Bucket (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)\n"
        "- **Type:** Multi-class classification (6 classes)"
    )

    with st.expander("Feature Details"):
        st.dataframe(pd.DataFrame({
            'Feature': ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene'],
            'Description': ['Fine particulate matter','Particulate matter','Nitric oxide','Nitrogen dioxide',
                           'Nitrogen oxides','Ammonia','Carbon monoxide','Sulphur dioxide','Ozone',
                           'Benzene','Toluene','Xylene'],
            'Unit': ['ug/m3','ug/m3','ug/m3','ug/m3','ppb','ug/m3','mg/m3','ug/m3','ug/m3','ug/m3','ug/m3','ug/m3']
        }), use_container_width=True, hide_index=True)

    with st.expander("AQI Categories"):
        st.dataframe(pd.DataFrame({
            'Bucket': ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe'],
            'AQI Range': ['0-50','51-100','101-200','201-300','301-400','401-500'],
            'Health Impact': ['Minimal','Minor discomfort (sensitive)','Lung/heart discomfort',
                             'Discomfort to most','Respiratory illness','Serious health impacts']
        }), use_container_width=True, hide_index=True)

    st.header("Models Used")
    c1, c2 = st.columns(2)
    c1.markdown("1. **Logistic Regression** (multinomial)\n2. **Decision Tree**\n3. **KNN**")
    c2.markdown("4. **Gaussian Naive Bayes**\n5. **Random Forest** (Ensemble)\n6. **XGBoost** (Ensemble)")

# ---- COMPARISON ----
elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    st.markdown("---")

    if not metrics_data:
        st.warning("No metrics yet. Run model/train_models.py first.")
    else:
        res = metrics_data['results']
        df = pd.DataFrame(res).T.reset_index()
        df.columns = ['Model'] + list(df.columns[1:])

        st.subheader("Metrics Table")
        st.dataframe(
            df.style.highlight_max(subset=['Accuracy','AUC','Precision','Recall','F1','MCC'], color='#90EE90')
                     .highlight_min(subset=['Accuracy','AUC','Precision','Recall','F1','MCC'], color='#FFB6C1')
                     .format({'Accuracy':'{:.4f}','AUC':'{:.4f}','Precision':'{:.4f}',
                              'Recall':'{:.4f}','F1':'{:.4f}','MCC':'{:.4f}'}),
            use_container_width=True, hide_index=True
        )
        best_name = df.loc[df['Accuracy'].idxmax(), 'Model']
        st.success(f"**Best (Accuracy):** {best_name} - {df['Accuracy'].max():.4f}")

        st.markdown("---")
        st.subheader("Charts")
        pick = st.selectbox("Metric:", ['All Metrics','Accuracy','AUC','Precision','Recall','F1','MCC'])

        fig, ax = plt.subplots(figsize=(12, 6))
        mets = ['Accuracy','AUC','Precision','Recall','F1','MCC']
        cols = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6','#1abc9c']

        if pick == 'All Metrics':
            x = np.arange(len(df))
            w = 0.12
            for i, (m, c) in enumerate(zip(mets, cols)):
                ax.bar(x+i*w, df[m], w, label=m, color=c, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x + w*2.5)
            ax.set_xticklabels(df['Model'], rotation=25, ha='right')
            ax.legend(loc='lower right')
        else:
            bars = ax.bar(df['Model'], df[pick], color=cols, edgecolor='black', linewidth=0.5)
            ax.set_xticklabels(df['Model'], rotation=25, ha='right')
            for b, v in zip(bars, df[pick]):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}',
                        ha='center', fontweight='bold', fontsize=10)

        ax.set_ylim(0, 1.15)
        ax.set_title(f'Model Comparison - {pick}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # radar
        st.subheader("Radar - Top 3")
        top3 = df.nlargest(3, 'Accuracy')
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2*np.pi, len(mets), endpoint=False).tolist()
        angles += angles[:1]
        rcols = ['#3498db','#e74c3c','#2ecc71']
        for i, (_, row) in enumerate(top3.iterrows()):
            vals = [row[m] for m in mets] + [row[mets[0]]]
            ax.plot(angles, vals, 'o-', linewidth=2, label=row['Model'], color=rcols[i])
            ax.fill(angles, vals, alpha=0.1, color=rcols[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(mets)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig)

# ---- PREDICTIONS ----
elif page == "Predictions":
    st.title("Make Predictions")
    st.markdown("---")

    if not models:
        st.error("No models loaded. Run model/train_models.py first.")
        st.stop()

    st.subheader("1. Pick a model")
    chosen = st.selectbox("Model:", list(models.keys()), index=len(models)-1)

    st.markdown("---")
    st.subheader("2. Upload test data (CSV)")
    st.info("Upload CSV with pollutant features. If 'AQI_Bucket' column exists, metrics will be shown.")

    c1, c2 = st.columns([3, 1])
    uploaded = c1.file_uploader("CSV file", type="csv")
    use_sample = c2.button("Use Sample Data")

    test_df = None
    if uploaded:
        test_df = pd.read_csv(uploaded)
        st.success(f"Loaded {uploaded.name} ({test_df.shape[0]} rows)")
    elif use_sample and sample_data is not None:
        test_df = sample_data.copy()
        st.success(f"Sample data loaded ({test_df.shape[0]} rows)")

    if test_df is not None:
        with st.expander("Preview"):
            st.dataframe(test_df.head(20), use_container_width=True)

        st.markdown("---")
        target_col = 'AQI_Bucket'
        has_target = target_col in test_df.columns
        feat_names = metrics_data['feature_names'] if metrics_data else \
            ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']

        missing = [c for c in feat_names if c not in test_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X_test = test_df[feat_names].copy()
        if X_test.isnull().sum().sum() > 0:
            X_test.fillna(X_test.median(), inplace=True)

        X_scaled = scaler.transform(X_test) if scaler else X_test.values

        mdl = models[chosen]
        y_pred_enc = mdl.predict(X_scaled)
        y_prob = mdl.predict_proba(X_scaled)

        if le:
            y_pred_labels = le.inverse_transform(y_pred_enc)
        elif metrics_data:
            cl = metrics_data['class_labels']
            y_pred_labels = [cl[i] for i in y_pred_enc]
        else:
            y_pred_labels = y_pred_enc

        st.subheader("3. Results")
        result = test_df.copy()
        result['Predicted_AQI'] = y_pred_labels
        result['Confidence'] = np.max(y_prob, axis=1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", len(y_pred_enc))
        top_pred = pd.Series(y_pred_labels).value_counts().index[0]
        c2.metric("Most Common", top_pred)
        c3.metric("Avg Confidence", f"{np.mean(np.max(y_prob, axis=1))*100:.1f}%")

        pred_counts = pd.Series(y_pred_labels).value_counts()
        pcols = st.columns(min(len(pred_counts), 6))
        for i, (label, cnt) in enumerate(pred_counts.items()):
            pcols[i % len(pcols)].metric(label, f"{cnt} ({cnt/len(y_pred_labels)*100:.1f}%)")

        with st.expander("All predictions", expanded=True):
            st.dataframe(result[['Predicted_AQI','Confidence'] + feat_names].style.format({'Confidence':'{:.4f}'}),
                         use_container_width=True)

        if has_target:
            y_true_labels = test_df[target_col]
            if le:
                try: y_true_enc = le.transform(y_true_labels)
                except: y_true_enc = y_pred_enc
            else:
                y_true_enc = y_true_labels

            st.markdown("---")
            st.subheader("4. Evaluation")

            acc = accuracy_score(y_true_enc, y_pred_enc)
            try: auc = roc_auc_score(y_true_enc, y_prob, multi_class='ovr', average='weighted')
            except: auc = 0.0
            prec = precision_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)
            rec = recall_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)
            f1v = f1_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_true_enc, y_pred_enc)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("AUC", f"{auc:.4f}")
            c3.metric("Precision", f"{prec:.4f}")
            c4.metric("Recall", f"{rec:.4f}")
            c5.metric("F1", f"{f1v:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")

            st.markdown("---")
            left, right = st.columns(2)

            with left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true_enc, y_pred_enc)
                cl = metrics_data['class_labels'] if metrics_data else sorted(set(y_true_labels))
                short = [l[:4] for l in cl]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=short, yticklabels=short,
                            linewidths=1, linecolor='black', annot_kws={'size': 12, 'fontweight': 'bold'})
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'{chosen}', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

            with right:
                st.subheader("Classification Report")
                tnames = metrics_data['class_labels'] if metrics_data else None
                try:
                    rpt = classification_report(y_true_enc, y_pred_enc, target_names=tnames,
                                                output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(rpt).T.style.format('{:.4f}'), use_container_width=True)
                except Exception as e:
                    st.error(f"Report error: {e}")
    else:
        st.info("Upload a CSV or click 'Use Sample Data' to start.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>India AQI Classification | ML Assignment 2 | BITS Pilani</div>",
            unsafe_allow_html=True)
