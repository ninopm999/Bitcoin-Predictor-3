    import streamlit as st
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from pycoingecko import CoinGeckoAPI
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    import plotly.graph_objs as go
    from datetime import datetime, timedelta
    import logging

    # Configure logging for debugging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Streamlit app configuration
    st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")

    # Title and description
    st.title("Bitcoin Price Predictor")
    st.markdown("""
    This app predicts Bitcoin prices using an LSTM model trained on historical data. Select a prediction horizon to view forecasted prices. Note: Predictions are for informational purposes only and not financial advice.
    """)

    # Function to fetch data from yfinance
    def fetch_yfinance_data(start_date, end_date):
        try:
            logger.info(f"Fetching yfinance data for BTC-USD from {start_date} to {end_date}")
            data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
            if data.empty:
                logger.warning("yfinance returned empty data")
                return None
            prices = data['Close'].values.reshape(-1, 1)
            logger.info(f"yfinance fetched {len(prices)} samples")
            return prices
        except Exception as e:
            logger.error(f"yfinance failed: {str(e)}")
            st.warning(f"yfinance failed: {str(e)}")
            return None

    # Function to fetch data from CoinGecko
    def fetch_coingecko_data(start_date, end_date):
        try:
            logger.info(f"Fetching CoinGecko data for bitcoin from {start_date} to {end_date}")
            cg = CoinGeckoAPI()
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            data = cg.get_coin_market_chart_range_by_id(
                id='bitcoin', vs_currency='usd', from_timestamp=start_timestamp, to_timestamp=end_timestamp
            )
            prices = [item[1] for item in data['prices']]
            if not prices:
                logger.warning("CoinGecko returned empty data")
                return None
            prices = np.array(prices).reshape(-1, 1)
            logger.info(f"CoinGecko fetched {len(prices)} samples")
            return prices
        except Exception as e:
            logger.error(f"CoinGecko failed: {str(e)}")
            st.warning(f"CoinGecko failed: {str(e)}")
            return None

    # Mock data as last resort for testing
    def fetch_mock_data():
        logger.info("Using mock data as fallback")
        st.warning("Using mock data due to failure in fetching real data. Predictions may not reflect real market conditions.")
        # Generate synthetic data (e.g., linear trend with noise)
        dates = pd.date_range(end=datetime.now(), periods=730, freq='D')
        prices = np.linspace(30000, 60000, 730) + np.random.normal(0, 1000, 730)
        return prices.reshape(-1, 1)

    # Combined data fetching function with fallback
    @st.cache_data
    def fetch_data(start_date, end_date):
        prices = fetch_yfinance_data(start_date, end_date)
        if prices is None or len(prices) == 0:
            logger.info("Falling back to CoinGecko API")
            st.info("Falling back to CoinGecko API for data retrieval.")
            prices = fetch_coingecko_data(start_date, end_date)
        if prices is None or len(prices) == 0:
            logger.info("Falling back to mock data")
            st.info("Falling back to mock data due to API failures.")
            prices = fetch_mock_data()
        if prices is None or len(prices) == 0:
            logger.error("All data fetching attempts failed")
            st.error("Failed to fetch Bitcoin price data from all sources. Please try again later.")
            st.stop()
        return prices

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

        # Debug: Display data shape
        logger.info(f"Prices shape: {prices.shape}")
        st.write(f"DEBUG: Fetched {prices.shape[0]} data points")

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
        try:
            cg = CoinGeckoAPI()
            historical_data = cg.get_coin_market_chart_range_by_id(
                id='bitcoin', vs_currency='usd',
                from_timestamp=int((datetime.now() - timedelta(days=100)).timestamp()),
                to_timestamp=int(datetime.now().timestamp())
            )
            historical_dates = [datetime.fromtimestamp(item[0] / 1000) for item in historical_data['prices'][-100:]]
            historical_prices = [item[1] for item in historical_data['prices'][-100:]]
        except Exception as e:
            logger.error(f"Failed to fetch visualization data from CoinGecko: {str(e)}")
            st.warning("Using mock data for visualization due to API failure.")
            historical_dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            historical_prices = prices[-100:].flatten()

        pred_dates = [datetime.now() + timedelta(days=i) for i in range(1, horizon + 1)]

        # Plot historical and predicted prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Prices'))
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
