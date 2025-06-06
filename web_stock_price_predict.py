import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.title("📈 Stock Price Prediction App")


stock = st.text_input("Enter the Stock ID", "GOOG")


end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)


google_data = yf.download(stock, start=start, end=end)


if google_data.empty:
    st.error("❌ No stock data found. Please enter a valid stock symbol.")
    st.stop()


model_path = "Latest_stock_price_model.keras"
try:
    model = load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


st.subheader("Stock Data")
st.write(google_data.tail(10))


google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()


def plot_graph(figsize, values, full_data, extr_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, label="Moving Average", color="red")
    plt.plot(full_data['Close'], label="Actual Prices", color="blue")
    plt.legend()
    if extr_data:
        plt.plot(extra_dataset)
    return fig


st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 & 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))


splitting_len = int(len(google_data) * 0.8)
x_test = google_data[['Close']][splitting_len:]


st.write(f"🔍 Total rows in dataset: {len(google_data)}")
st.write(f"🔎 Rows in x_test (after 80% split): {len(x_test)}")

if x_test is None or x_test.empty or len(x_test[['Close']]) < 101:
    st.error("❌ Not enough data after the 80% split to make predictions. Try another stock with more historical data.")
    st.stop()

# Preprocessing
# Preprocessing
scaler = StandardScaler()


x_close = x_test[['Close']].dropna().copy()


if x_close.empty or len(x_close) < 101:
    st.error(f"❌ Not enough usable 'Close' values after dropna(). Got {len(x_close)} rows.\n"
             f"Try a stock with longer history or fewer missing values.")
    st.write("🔎 x_test shape:", x_test.shape)
    st.write("🔍 x_close preview:", x_close.tail())
    st.stop()


scaled_data = scaler.fit_transform(x_close)


x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)


predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)


ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_predictions.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([
    google_data['Close'][:splitting_len + 100],
    ploting_data['original_test_data'],
    ploting_data['predictions']
], axis=1))
plt.legend(["Training Data", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)


last_100_days = google_data['Close'][-100:].values.reshape(-1, 1)
scaled_last_100_days = scaler.transform(last_100_days)
X_next_day = np.array([scaled_last_100_days])  # Shape (1, 100, 1)

predicted_next_day_scaled = model.predict(X_next_day)
predicted_next_day_price = scaler.inverse_transform(predicted_next_day_scaled)
next_day_date = end.date() + timedelta(days=1)


st.subheader("Predicted Stock Price for Next Day")
st.write(f"📈 **Predicted price for {next_day_date}:** {predicted_next_day_price[0][0]:.2f}  \n"
         f"💡 For foreign stocks: USD ($), for Indian stocks: INR (₹)")


st.subheader("Last 100 Days Candlestick Chart with Predicted Next Day Price")
candlestick_data = google_data[-100:].copy().reset_index()
candlestick_data['Date'] = candlestick_data['Date'].dt.strftime('%Y-%m-%d')

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=candlestick_data['Date'],
    open=candlestick_data['Open'],
    high=candlestick_data['High'],
    low=candlestick_data['Low'],
    close=candlestick_data['Close'],
    name='Candlestick Chart'
))


fig.add_trace(go.Scatter(
    x=[next_day_date.strftime('%Y-%m-%d')],
    y=[predicted_next_day_price[0][0]],
    mode='markers',
    marker=dict(color='red', size=10),
    name='Predicted Price'
))

fig.update_layout(title=f"{stock} - Last 100 Days with Next Day Prediction",
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  xaxis_rangeslider_visible=False)

st.plotly_chart(fig)
