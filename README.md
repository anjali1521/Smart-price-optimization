# 🧠 Smart Price Optimization using XGBoost

## The Problem We Solve

**Pricing is the toughest decision for any business.**

Imagine a shop selling noodles. A few years ago when there was no competition, they could charge ₹5. Then as the market grew, they increased price to ₹10. Customers accepted it because:
- The brand was trusted
- Quality was good
- No better alternatives existed

**Then competition arrived.**

Other brands offered similar noodles at lower prices. The shop thought: "We're the market leader, let's charge ₹12 instead."

**What happened?** Customers switched to competitors offering the same product at ₹10. Sales crashed. Profit collapsed.

**Why?** Because when competition exists, you can't just increase price. Customers will leave.

---

## 💡 The Real Challenge

This happens to EVERY business:

✗ **Too high price** → Customers switch to competitors → Low volume → Low profit  
✗ **Too low price** → Volume is good but margins crushed → Low profit  
✓ **Perfect price** → Good volume + Good margins = Maximum profit  

**But how do you find that perfect price?**

Most businesses guess. They try ₹10, see what sells. Then try ₹12, see sales drop. Then go back to ₹10. This is chaos!

**What if you could predict the best price using DATA?**

---

## 🚀 What This Project Does

This system uses **Machine Learning (XGBoost)** to analyze your past sales data and tell you:

- "Your optimal price is ₹10.50, not ₹12"
- "This product can handle a price increase"
- "This product should keep its current price"
- "If competitor goes down to ₹8, you should reconsider"

### Real Impact:

```
BEFORE (Guessing):
Month 1: ₹10 price, ₹3,000 profit
Month 2: Increase to ₹12, ₹1,000 profit (oops!)
Month 3: Back to ₹10, ₹3,000 profit

Average: ₹2,333/month (wasted potential)

AFTER (Smart Pricing):
Month 1: ₹10 price, ₹3,000 profit
Month 2: Tested ₹10.50, ₹3,200 profit ✅
Month 3: Stayed at ₹10.50, ₹3,200 profit ✅

Average: ₹3,133/month (30% GAIN!)
```

---

## 📊 How It Works

### Step 1: Feed Your Historical Data
```
Collect your past sales records:
- What price you charged
- How many units sold at that price
- Your cost per unit
- Date/time period
```

### Step 2: ML Model Learns Patterns
```
The XGBoost model analyzes:
- "When price went from ₹10 to ₹12, sales dropped 80%"
- "When price stayed at ₹10, sales stayed consistent"
- "Customers are price-sensitive to changes above ₹11"
```

### Step 3: Get Recommendations
```
Model tells you:
- Current price: ₹10, Sales: 100/day, Profit: ₹200/day
- If you try ₹10.50: Predicted sales: 95/day, Profit: ₹220/day ✅
- If you try ₹11: Predicted sales: 40/day, Profit: ₹80/day ❌
- If you try ₹12: Predicted sales: 10/day, Profit: ₹30/day ❌

Recommendation: Stay at ₹10 or try ₹10.50 MAX
```

### Step 4: Make Data-Backed Decisions
```
No more guessing!
- Implement the recommended price
- Monitor sales
- Adjust based on real market conditions
```

---

## 📁 Dataset Format

Place `retail_price.csv` in the project folder with these columns:

| Column | Description |
|--------|-------------|
| `unit_price` | Price you're currently charging (₹) |
| `units_sold` | Number of units sold at that price |
| `cost_price` | Cost to produce/buy the unit |
| `product_category_name` | Category (Noodles, Snacks, etc.) |
| `product_id` | Unique product identifier |
| `month_year` | Month of sale (for trends) |

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn xgboost joblib streamlit scikit-learn
```

Or use:
```bash
pip install -r requirements.txt
```

---

## 🧪 Model Details

| Aspect | Details |
|--------|---------|
| **Algorithm** | XGBoost Regressor |
| **Purpose** | Predict optimal pricing |
| **Training Data** | 80% of historical sales |
| **Testing Data** | 20% for validation |
| **Metrics** | RMSE (accuracy) & R² (model quality) |
| **Parameters** | max_depth=5, learning_rate=0.1, n_estimators=150 |

---

## 🎯 Dashboard Features

When you run the app, you get:

1. **📊 Data Preview** - See your sales history at a glance
2. **📈 Model Performance** - How accurate are the predictions?
3. **🔍 Feature Importance** - What factors matter most for pricing?
4. **💡 Top Opportunities** - Which products are underpriced/overpriced?
5. **🧾 Custom Prediction** - "What if I price this at ₹12?" Get instant predictions
6. **📉 Price vs Sales Relationship** - Visualize how price affects demand

---

## 💻 How to Run

```bash
streamlit run smart_price_optimization.py
```

Then:
1. Upload your sales data (or use the included retail_price.csv)
2. View the dashboard
3. Check predictions for different price points
4. Get recommendations
5. Implement the optimal price

---

## 📈 Real-World Use Cases

✅ **Retail shops** - Price products to beat competition while keeping profit  
✅ **Food vendors** - Find the price customers will happily pay  
✅ **E-commerce stores** - Dynamic pricing based on market data  
✅ **Restaurants** - Menu item pricing strategy  
✅ **FMCG brands** - Wholesale pricing optimization  

---

## 🎓 Key Learning

### The Sweet Spot Formula

```
PROFIT = (Price - Cost) × Quantity Sold

NOT: "Higher price = more profit"
BUT: "Price that keeps volume high = more profit"

EXAMPLE:
₹12 price: (12-6) × 10 units = ₹60 profit
₹10 price: (10-6) × 50 units = ₹200 profit ✅
