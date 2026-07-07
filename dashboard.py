# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# --- Cache model loading (runs only once) ---
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\kanan\OneDrive\Documents\ml ai internship\project\best_rf_model.pkl")

model = load_model()

# Feature list (matches your dataset)
features = [
    "Household_ID_encoded", "Household_Size", "Avg_Temperature_C", "Has_AC",
    "Peak_Hours_Usage_kWh", "day_of_week", "month",
    "is_weekend", "lag_1", "lag_7"
]

st.title("Energy Consumption Prediction (Random Forest)")

# --- User Input Section ---
household_id = st.number_input("Household ID (encoded)", min_value=0, value=1)
household_size = st.number_input("Household Size", min_value=1, value=3)
avg_temp = st.number_input("Average Temperature (°C)", value=25.0)
has_ac = st.selectbox("Has AC?", [0, 1])
peak_usage = st.number_input("Peak Hours Usage (kWh)", value=10.0)
day_of_week = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=7)
is_weekend = st.selectbox("Is Weekend?", [0, 1])
lag_1 = st.number_input("Lag 1 (previous day consumption)", value=50.0)
lag_7 = st.number_input("Lag 7 (last week consumption)", value=55.0)

if st.button("Predict"):
    input_data = np.array([[household_id, household_size, avg_temp, has_ac,
                            peak_usage, day_of_week, month,
                            is_weekend, lag_1, lag_7]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")

# --- Evaluation Section ---
st.header("Model Evaluation & Graphs")

# Replace with your actual test dataset
# Example: load from CSV or split dataset
X_test = np.random.randint(1, 12, (200, len(features)))  # placeholder
y_test = np.random.randint(40, 70, 200)                 # placeholder

y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.3f}")
st.write(f"MAPE: {mape*100:.2f}%")

# --- Graphs ---
# 1. Actual vs Predicted
st.subheader("Actual vs Predicted")
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, color="blue", alpha=0.6, s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
st.pyplot(fig1)

# 2. Residual Plot
st.subheader("Residuals vs Predictions")
fig2, ax2 = plt.subplots()
ax2.scatter(y_pred, residuals, color="green", alpha=0.6, s=20)
ax2.axhline(y=0, color="red", linestyle="--")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")
st.pyplot(fig2)

# 3. Residual Distribution
st.subheader("Residual Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(residuals, bins=15, kde=True, color="purple", ax=ax3)
ax3.axvline(x=0, color="red", linestyle="--")
ax3.set_xlabel("Residuals")
st.pyplot(fig3)

# 4. Feature Importance
st.subheader("Feature Importance")
importances = model.feature_importances_
feat_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

fig4, ax4 = plt.subplots()
ax4.barh(feat_imp["Feature"], feat_imp["Importance"], color="skyblue")
ax4.set_xlabel("Importance")
st.pyplot(fig4)

st.write(feat_imp)

# 5. Feature Importance Heatmap (compact)
st.subheader("Feature Importance Heatmap")
feat_imp_matrix = pd.DataFrame(importances.reshape(1, -1), columns=features)
fig5, ax5 = plt.subplots(figsize=(12, 2))  # wide, short
sns.heatmap(feat_imp_matrix, annot=True, cmap="coolwarm", ax=ax5, cbar=False)
ax5.set_title("Feature Importance Heatmap")
st.pyplot(fig5)

# --- Download Predictions & Residuals ---
st.subheader("Download Predictions & Residuals")
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residuals": residuals
})
csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name="predictions_residuals.csv",
    mime="text/csv"
)
