import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Streamlit app configuration
st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")

# Title and description
st.title("Bitcoin Price Predictor")
st.markdown("""
This app predicts Bitcoin prices using an LSTM model trained on historical data. Select a prediction horizon to view forecasted prices. Note: Predictions are for informational purposes only and not financial advice.
""")

# Function to fetch and preprocess data
@st.cache_data
def fetch_data(start_date, end_date):
    data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    return data['Close'].values.reshape(-1, 1)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Function to train LSTM model
@st.cache_resource
def train_model(X_train, y_train, seq_length):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

# Function to make predictions
def predict_prices(model, scaler, recent_data, seq_length, horizon):
    predictions = []
    input_seq = recent_data[-seq_length:].reshape(1, seq_length, 1)
    for _ in range(horizon):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.roll(input_seq, -1, axis=1)
        input_seq[0, -1, 0] = pred
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Main app logic
def main():
    # User input for prediction horizon
    horizon = st.slider("Select Prediction Horizon (days)", 1, 30, 7)

    # Fetch historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    with st.spinner("Fetching Bitcoin price data..."):
        prices = fetch_data(start_date, end_date)

    # Preprocess data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    seq_length = 60
    X, y = create_sequences(prices_scaled, seq_length)
    
    # Train model
    with st.spinner("Training the model..."):
        model = train_model(X, y, seq_length)

    # Make predictions
    with st.spinner("Generating predictions..."):
        pred_prices = predict_prices(model, scaler, prices_scaled, seq_length, horizon)

    # Prepare data for visualization
    last_date = pd.to_datetime(yf.download('BTC-USD', start=start_date, end=end_date).index[-1])
    pred_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    historical_dates = pd.to_datetime(yf.download('BTC-USD', start=start_date, end=end_date).index[-100:])
    historical_prices = prices[-100:]

    # Plot historical and predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices.flatten(), mode='lines', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices.flatten(), mode='lines', name='Predicted Prices', line=dict(dash='dash')))
    fig.update_layout(title="Bitcoin Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Display predictions
    st.subheader("Predicted Prices")
    pred_df = pd.DataFrame({"Date": pred_dates, "Predicted Price (USD)": pred_prices.flatten()})
    st.dataframe(pred_df.style.format({"Predicted Price (USD)": "{:.2f}"}))

    # Disclaimer
    st.markdown("""
    **Disclaimer**: This app uses machine learning to predict Bitcoin prices based on historical data. The accuracy is not guaranteed, and cryptocurrency investments carry high risks. Use at your own discretion.
    """)

if __name__ == "__main__":
    main()
