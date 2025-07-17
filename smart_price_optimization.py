import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smart Price Optimizer", layout="wide")

st.title("🧠 Smart Price Optimization using XGBoost")
st.markdown("Optimize your product pricing using machine learning and advanced analytics.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("retail_price.csv")
    df.drop(columns=["product_id", "month_year"], inplace=True)
    df = pd.get_dummies(df, columns=["product_category_name"], drop_first=True)
    return df

df = load_data()
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# Split data
target = "unit_price"
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# Feature importance
st.subheader("🔍 Feature Importance")
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, ax=ax, max_num_features=15)
st.pyplot(fig)

# Show top pricing opportunities
recommendation_df = X_test.copy()
recommendation_df["Predicted Price"] = y_pred
recommendation_df["Actual Price"] = y_test.values
recommendation_df["Delta"] = recommendation_df["Predicted Price"] - recommendation_df["Actual Price"]
top_deltas = recommendation_df.sort_values(by="Delta", ascending=False).head(5)

st.subheader("💡 Top 5 Price Optimization Opportunities")
st.dataframe(top_deltas[["Predicted Price", "Actual Price", "Delta"]])

# Predict price for custom input
st.subheader("🧾 Predict Price for Custom Input")

input_data = {}
for col in X.columns:
    if df[col].dtype == "float64" or df[col].dtype == "int64":
        input_data[col] = st.number_input(col, value=float(df[col].mean()))
    else:
        input_data[col] = st.selectbox(col, df[col].unique())

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    pred_price = model.predict(input_df)[0]
    st.success(f"✅ Predicted Unit Price: ₹{pred_price:.2f}")

# Save model (optional)
joblib.dump(model, "xgboost_price_model.pkl")