# India Air Quality Index (AQI) Classification using ML Models

**Name:** Harsh Deep Singh
**Roll No:** 2025ab05204
**Email:** 2025ab05204@wilp.bits-pilani.ac.in

## Problem Statement

Air pollution is a serious public health problem in India. This project uses six ML classification models to classify air quality into AQI categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) based on pollutant concentration data collected from monitoring stations across India between 2015-2020. The models are compared on standard metrics and deployed via a Streamlit web app.

## Dataset Description

- **Name:** Air Quality Data in India (2015-2020)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Type:** Multi-class Classification (6 classes)
- **Features:** 12 pollutant measurements
- **Instances:** 24,365 (after cleaning)
- **Target:** `AQI_Bucket`

### Features

| Feature | Description | Unit |
|---------|-------------|------|
| PM2.5 | Fine particulate matter | ug/m3 |
| PM10 | Particulate matter | ug/m3 |
| NO | Nitric oxide | ug/m3 |
| NO2 | Nitrogen dioxide | ug/m3 |
| NOx | Nitrogen oxides | ppb |
| NH3 | Ammonia | ug/m3 |
| CO | Carbon monoxide | mg/m3 |
| SO2 | Sulphur dioxide | ug/m3 |
| O3 | Ozone | ug/m3 |
| Benzene | Benzene | ug/m3 |
| Toluene | Toluene | ug/m3 |
| Xylene | Xylene | ug/m3 |

### Target Classes

| AQI Bucket | Range | Health Impact |
|---|---|---|
| Good | 0-50 | Minimal |
| Satisfactory | 51-100 | Minor discomfort to sensitive people |
| Moderate | 101-200 | Discomfort to people with lung/heart disease |
| Poor | 201-300 | Discomfort to most people |
| Very Poor | 301-400 | Respiratory illness on prolonged exposure |
| Severe | 401-500 | Serious health impacts |

## Models and Evaluation Metrics

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7640 | 0.9355 | 0.7636 | 0.7640 | 0.7594 | 0.6751 |
| Decision Tree | 0.7593 | 0.8630 | 0.7585 | 0.7593 | 0.7587 | 0.6725 |
| KNN | 0.7732 | 0.9331 | 0.7712 | 0.7732 | 0.7714 | 0.6902 |
| Naive Bayes | 0.6518 | 0.8865 | 0.6623 | 0.6518 | 0.6540 | 0.5343 |
| Random Forest (Ensemble) | 0.8184 | 0.9623 | 0.8176 | 0.8184 | 0.8171 | 0.7519 |
| XGBoost (Ensemble) | 0.8196 | 0.9599 | 0.8187 | 0.8196 | 0.8189 | 0.7542 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.7640, AUC 0.9355. Multinomial handles multi-class okay but linear boundaries miss pollutant interactions. |
| Decision Tree | Accuracy 0.7593, MCC 0.6725. Good at non-linear pollutant thresholds. Depth limiting prevents overfitting. |
| KNN | Accuracy 0.7732, F1 0.7714. Works decently after scaling. Weighted distances help. |
| Naive Bayes | Accuracy 0.6518, AUC 0.8865. Gaussian assumption rough fit for pollutant data. |
| Random Forest (Ensemble) | Accuracy 0.8184, AUC 0.9623. Bagging works well - captures pollutant interactions. |
| XGBoost (Ensemble) | Accuracy 0.8196, MCC 0.7542. Strong across all metrics. Handles class imbalance well. |

## Project Structure

```
ML_Assignment2_B/
|-- app.py                  # Streamlit web application
|-- requirements.txt        # Python dependencies
|-- README.md               # This file
|-- model/
    |-- train_models.ipynb  # Model training notebook
    |-- *.pkl               # Saved model files (generated after training)
    |-- metrics.json        # Evaluation metrics (generated after training)
    |-- test_sample.csv     # Sample test data (generated after training)
```

## How to Run

### 1. Train the models
1. Open `model/train_models.ipynb` in Jupyter / VS Code
2. Make sure `city_day.csv` is in the `model/` folder
3. Run all cells
4. Model files, metrics, and plots will be saved in `model/`

### 2. Run the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub, select repo, pick `app.py`, deploy

## Links

- **GitHub Repository:** https://github.com/HastroSingh/India_AQI_Classification
- **Live Streamlit App:** https://harshsinghindiaaqiclassification.streamlit.app/

## Screenshots

> BITS Virtual Lab screenshot goes here
