%%writefile app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 Stock Price Predictor with LSTM")
stock = st.sidebar.text_input("Enter Stock Symbol", value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-12-31'))

@st.cache_data
def load_data(stock):
    df = yf.download(stock, start=start_date, end=end_date)
    return df

df = load_data(stock)
st.subheader(f"{stock} Closing Price Data")
st.line_chart(df['Close'])

# Scale data
data = df[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare training data
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i - time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
with st.spinner('Training the model...'):
    model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=0)

# Test prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 📈 Plot actual vs predicted
st.subheader("📊 Actual vs Predicted Prices")
fig1, ax1 = plt.subplots()
ax1.plot(actual, label='Actual')
ax1.plot(predictions, label='Predicted')
ax1.legend()
st.pyplot(fig1)

# 🧮 Predict next 7 days
st.subheader("📅 Next 7 Days Forecast")
last_60_days = scaled_data[-60:].reshape(1, -1)
temp_input = list(last_60_days[0])
next_7_days = []

for i in range(7):
    x_input = np.array(temp_input[-60:]).reshape(1, 60, 1)
    pred = model.predict(x_input, verbose=0)
    next_7_days.append(pred[0][0])
    temp_input.append(pred[0][0])

next_7_days_unscaled = scaler.inverse_transform(np.array(next_7_days).reshape(-1, 1))

# Show prediction
next_7_df = pd.DataFrame(next_7_days_unscaled, columns=["Predicted Close"])
next_7_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
st.dataframe(next_7_df)

# Plot forecast
fig2, ax2 = plt.subplots()
ax2.plot(next_7_df.index, next_7_df["Predicted Close"], marker='o', linestyle='-')
ax2.set_title("Next 7 Days Forecast")
st.pyplot(fig2)

