# 📈 Stock Market Prediction 

This project is a **Stock Market Price Prediction App** built with **Python, Keras (LSTM model), and Streamlit**.  
It predicts stock prices based on historical data and provides visual insights with moving averages.  

---

## 🚀 Features
- Fetches real-time stock data using **Yahoo Finance API (yfinance)**.  
- Data visualization with **Moving Averages (50, 100, 200 days)**.  
- Price prediction using a pre-trained **LSTM deep learning model**.  
- Displays:  
  - Stock data table  
  - MA50, MA100, MA200 comparisons  
  - Original vs Predicted prices  
  - Next-day predicted closing price  

---

## 🛠️ Tech Stack
- **Python**  
- **TensorFlow / Keras** (LSTM model)  
- **Streamlit** (interactive web app)  
- **Pandas / NumPy** (data processing)  
- **Matplotlib** (data visualization)  
- **yfinance** (stock data fetching)  

---

## 📷 Predicted Stock Results

### 1️⃣ Stock Data Table  
![Stock Data](./Screenshot%202025-09-21%20101304.png)  

### 2️⃣ Original vs Predicted Prices  
![Original vs Predicted](./Screenshot%202025-09-21%20101242.png)  

### 3️⃣ Tomorrow’s Predicted Price  
![Tomorrow Prediction](./Screenshot%202025-09-21%20101353.png)  

---

## ⚡ How It Works
1. Enter a stock ticker (e.g., `RELIANCE.NS`, `AAPL`, `TSLA`) in the app.  
2. The app downloads stock data from **Yahoo Finance**.  
3. Data is preprocessed and scaled using **MinMaxScaler**.  
4. A trained **LSTM model** predicts future stock prices.  
5. Results are plotted and displayed on the Streamlit dashboard.  

---


