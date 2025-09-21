import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model(r'C:\Users\siddh\OneDrive\Desktop\Stock\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symbol You Need to predict', 'RELIANCE.NS')
start = '2012-01-01'
end = '2025-09-19'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

predict = scaler.inverse_transform(predict)
y = scaler.inverse_transform(y.reshape(-1, 1))

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

last_100_days = data_test_scale[-100:]   # last 100 days (scaled)
last_100_days = np.array(last_100_days).reshape(1, 100, 1)

# Predict next day (still scaled)
tomorrow_pred = model.predict(last_100_days)

# Inverse transform to get real price
tomorrow_pred = scaler.inverse_transform(tomorrow_pred)

# Show result
st.subheader("🔮 Predicted Price for Tomorrow")
st.write(f"Predicted Closing Price: **{tomorrow_pred[0][0]:.2f}**")