import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

# Streamlit Title
st.title("Stock Price Prediction & Visualization")

# User input for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., GOOG for Google):", "GOOG")

# Fetch stock data
def get_stock_data(symbol):
    try:
        data = yf.download(symbol, period="1mo", interval="1d")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Fetch data
google_data = get_stock_data(stock_symbol)

# ✅ Ensure google_data is a valid DataFrame
if google_data is None or google_data.empty:
    st.error("⚠️ Error: No stock data retrieved. Check the stock symbol and try again.")
    st.stop()

# 📌 Convert necessary columns to numeric format (only if they exist)
columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

for col in columns_to_convert:
    if col in google_data.columns:
        if isinstance(google_data[col], pd.Series):  # ✅ Ensure it's a Pandas Series
            google_data[col] = pd.to_numeric(google_data[col], errors='coerce')
        else:
            st.warning(f"⚠️ Column {col} is not a valid Series. Skipping conversion.")
    else:
        st.warning(f"⚠️ Column {col} not found in the data. Skipping.")

# ✅ Replace NaN values with sequential dummy values (0,1,2,3,4,...) for problematic columns
google_data.fillna({col: i for i, col in enumerate(columns_to_convert)}, inplace=True)

# 📌 Debugging: Show final data
st.write("✅ Data Types After Conversion:")
st.write(google_data.dtypes)

# 📊 Plot Candlestick Chart
st.subheader("Stock Candlestick Chart")
fig, ax = plt.subplots(figsize=(10, 5))
mpf.plot(google_data, type='candle', style='charles', ax=ax, ylabel='Price ($)')
st.pyplot(fig)

# ✅ Done! 🎉
