# 🧠 Smart Price Optimization using XGBoost

This is a machine learning-powered web app built with **Streamlit** that helps optimize product pricing based on historical data. It uses **XGBoost** for regression modeling and provides an interactive dashboard to visualize pricing insights and make custom predictions.

---

## 🚀 Features

- Load and clean pricing dataset (`retail_price.csv`)
- Encode categorical features with one-hot encoding
- Train and evaluate an XGBoost Regressor model
- Display model performance metrics: RMSE and R²
- Visualize feature importance
- Identify top 5 price optimization opportunities
- Custom input form for price prediction
- Save trained model with Joblib

---

## 📁 Dataset Format

Place `retail_price.csv` in the root project directory. It should contain:
- `unit_price` (target variable)
- `product_id`, `month_year`, `product_category_name` (feature columns)
- Other numerical and categorical features

---

## 📦 Requirements

Install required libraries using:

```bash
pip install pandas numpy matplotlib seaborn xgboost joblib streamlit scikit-learn
```
Or create a requirements.txt and run:

```bash
pip install -r requirements.txt
```
🧪 Model Overview
Algorithm: XGBoost Regressor

Objective: Predict unit prices based on product features

Metrics: RMSE and R² Score

Train/Test Split: 80% train / 20% test


🎯 Streamlit Dashboard
View data preview in table format

See RMSE and R² score in metric cards

Visualize feature importances with bar plot

Predict price for custom inputs using sidebar form

Display top 5 predictions with largest positive delta (underpricing)

💻 How to Run
Run the app with:

```bash
streamlit run app.py
```
Make sure your dataset is placed correctly and all libraries are installed.


🔐 Saving the Model
The trained model is saved as:

```bash
xgboost_price_model.pkl
```
You can reload it later to make predictions without retraining.

✨ Author
Developed by Anjali. Feel free to fork the project, give it a star 🌟, or open an issue with suggestions!
