# futuregridAI
# Energy Consumption Prediction Dashboard

A Streamlit web application for predicting household energy consumption using a tuned Random Forest model.  
This project demonstrates **feature engineering, model training, evaluation, and interactive visualization**.

---

## 🚀 Features
- **Interactive Inputs**: Household size, temperature, AC usage, peak hours, day of week, month, weekend flag, lag features.
- **Prediction Output**: Instant energy consumption forecast (kWh).
- **Evaluation Metrics**: RMSE, R², MAPE.
- **Visualizations**:
  - Actual vs Predicted
  - Residual plots
  - Residual distribution
  - Feature importance (bar chart + heatmap)
- **Download Option**: Export predictions & residuals as CSV.

---

## 🛠️ Tech Stack
- **Python 3.11+**
- **Libraries**:
  - Streamlit
  - Pandas
  - NumPy
  - Scikit‑learn
  - Matplotlib
  - Seaborn
  - Joblib

---

## 📂 Project Structure
project/
│── dashboard.py            # Streamlit app
│── train_model.py          # Model training script
│── best_rf_model.pkl       # Saved Random Forest model
│── household_energy_consumption.csv  # Dataset
│── requirements.txt        # Dependencies
│── README.md               # Project documentation



## ⚡ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
streamlit run dashboard.py

## ⚡ Dataset 
The dataset contains household energy consumption records with engineered features:

Household_ID_encoded

Household_Size

Avg_Temperature_C

Has_AC

Peak_Hours_Usage_kWh

day_of_week

month

is_weekend

lag_1

lag_7

Target: Energy_Consumption_kWh
