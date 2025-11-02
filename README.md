# Gold Prize Prediction

## 📌 Overview
This project predicts the **next 1-hour candle direction** (Up/Down) for **XAU/USD** and **GBP/USD** using a **Machine Learning model (Random Forest Classifier)** trained on historical intraday (1-hour) data fetched via the **Twelve Data API**.

The goal is to forecast whether the next candle will close **higher (Up)** or **lower/flat (Down)** compared to the current close, using technical indicators and historical patterns.

---

## ⚙️ Project Workflow

### 1. Data Collection (via Twelve Data API)
The project uses the **free Twelve Data API** to collect historical 1-hour OHLCV data for multiple forex pairs.

```python
def fetch_twelvedata(symbol, interval, outputsize, api_key):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
        "dp": 5
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'values' not in data:
        print(f"Error fetching {symbol}: {data.get('message', 'Unknown error')}")
        return None

    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df.set_index('datetime', inplace=True)
    df = df.astype(float)
    return df
```

**Parameters:**
- `symbol`: Forex pair (e.g., `'XAU/USD'`, `'GBP/USD'`)
- `interval`: Timeframe (e.g., `'1h'`)
- `outputsize`: Number of data points (max 5000)
- `api_key`: Your Twelve Data API key

After fetching, each dataset is saved as:
```
XAUUSD_1h.csv
GBPUSD_1h.csv
```

---

## 📊 Dataset Description

Each CSV file contains:

| Column | Description |
|--------|--------------|
| `datetime` | Time of the 1-hour candle |
| `open` | Opening price |
| `high` | Highest price |
| `low` | Lowest price |
| `close` | Closing price |
| `volume` | Trading volume (if available) |

From this, additional **derived indicators** and **features** are generated for training.

---

## 🧮 Feature Engineering

To make the dataset suitable for ML prediction, several **technical and statistical features** are calculated:

| Feature | Description |
|----------|--------------|
| `return` | Percentage change in closing price |
| `return_lag_1` ... `return_lag_5` | Past 5 returns to capture recent momentum |
| `SMA_10`, `SMA_50` | Short-term and medium-term moving averages |
| `rolling_std_10`, `rolling_std_50` | 10- and 50-period rolling volatility |
| `RSI_14` | Relative Strength Index (RSI) with 14-period smoothing |

**Target Variable:**  
`target = 1` if next close > current close, else `0` (Down/Flat).

---

## 🧠 Machine Learning Model

### Algorithm: **Random Forest Classifier**
A Random Forest ensemble model is used to classify whether the next 1-hour candle will close higher or not.

```python
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)
```

### Data Split:
- **80%** → Training Set  
- **20%** → Testing Set  
(Time-based split, preserving sequence)

### Feature Scaling:
`StandardScaler` ensures features have zero mean and unit variance for balanced model input.

---

## 📈 Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **Accuracy** | Percentage of correctly predicted directions |
| **ROC-AUC** | Measures ability to distinguish between Up and Down |
| **Classification Report** | Precision, Recall, F1-Score |
| **Confusion Matrix** | Breakdown of true vs predicted classes |

---

## 📉 Visualization

### 1. Actual vs Predicted Candle Directions
Shows how well predictions align with true market direction.

### 2. Probability of “Up” Predictions
Displays model confidence (probability output).

---

## 🔍 Feature Importance

The model ranks the most influential indicators:

```python
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.head(10))
```

Typical top features:
```
SMA_10
RSI_14
return_lag_1
SMA_50
rolling_std_10
```

---

## 🔮 Predicting the Next Hour Candle

The last available data row is used to forecast the upcoming candle’s direction:

```python
last_row = X.iloc[-1:]
last_row_s = scaler.transform(last_row)
next_prob = clf.predict_proba(last_row_s)[0,1]
next_pred = clf.predict(last_row_s)[0]
print(f"Next 1-hour candle prediction: {'UP' if next_pred==1 else 'DOWN/FLAT'} (prob_up={next_prob:.3f})")
```

**Output Example:**
```
Next 1-hour candle prediction: UP (prob_up=0.537)
```

---


Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone this repository or upload files to Google Colab.
2. Add your **Twelve Data API key**:
   ```python
   API_KEY = "YOUR_API_KEY"
   ```
3. Fetch data (optional) or use existing CSVs.
4. Run the training and evaluation cell.
5. View plots, metrics, and feature importance.
6. Check next-hour prediction output.

---

## 🧠 Future Improvements
- Add more technical indicators (MACD, Bollinger Bands, ATR)
- Implement time-series cross-validation
- Try deep learning (LSTM, GRU) for sequential pattern recognition
- Integrate real-time data streaming for live signal generation

---

## 📬 Author
**Rahul Patil**  
Machine Learning & Algorithmic Trading Enthusiast  
📍 India 
