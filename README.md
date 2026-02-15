# India Air Quality Index (AQI) Classification using ML Models

## Problem Statement

Air pollution is a serious public health problem in India. This project uses six ML classification models to classify air quality into AQI categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) based on pollutant concentration data collected from monitoring stations across India between 2015-2020. The models are compared on standard metrics and deployed via a Streamlit web app.

## Dataset Description

- **Name:** Air Quality Data in India (2015-2020)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Type:** Multi-class Classification (6 classes)
- **Features:** 12 pollutant measurements
- **Instances:** ~24,000+ (after cleaning)
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
| Logistic Regression | _run train_models.ipynb_ | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| XGBoost (Ensemble) | | | | | | |

> Run `model/train_models.ipynb` on Google Colab to get actual values.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | _Fill after training_ |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| XGBoost (Ensemble) | |

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

### 1. Train the models (Google Colab)
1. Open `model/train_models.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `city_day.csv` when prompted
3. Run all cells (~30-40 seconds)
4. Download the generated `.pkl`, `.json`, and `.csv` files
5. Place them in the `model/` folder

### 2. Run the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub, select repo, pick `app.py`, deploy

## Live App

> [Streamlit App Link - TBD]()

## Screenshots

> BITS Virtual Lab screenshot goes here
