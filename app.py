import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS  # Import CORS
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras import backend as K
import os

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)

# --- Data Loading ---
# Create a dictionary to map stock symbols to their CSV file paths
# Assumes CSVs are in the same directory as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = {
    "GOOGL": os.path.join(BASE_DIR, "GOOGL_2006-01-01_to_2018-01-01.csv"),
    "MSFT": os.path.join(BASE_DIR, "MSFT_2006-01-01_to_2018-01-01.csv"),
    "IBM": os.path.join(BASE_DIR, "IBM_2006-01-01_to_2018-01-01.csv"),
    "AMZN": os.path.join(BASE_DIR, "AMZN_2006-01-01_to_2018-01-01.csv")
}

# Cache for loaded data to avoid reading from disk every time
data_cache = {}

def load_data(stock_symbol):
    """Loads and caches stock data from CSV."""
    if stock_symbol in data_cache:
        return data_cache[stock_symbol]
    
    filepath = DATA_FILES.get(stock_symbol)
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found for symbol: {stock_symbol}")
        
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']] # We only care about the 'Close' price
    df = df.dropna()
    data_cache[stock_symbol] = df
    return df

# --- Model Functions ---

def get_arima_forecast(data, n_forecast=30):
    """Trains a simple ARIMA model and returns forecast."""
    # A simple (p,d,q) order. This should be properly tuned.
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_forecast)
    return forecast.tolist()

# Helper function to create sequences for GRU
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def get_gru_forecast(data, n_forecast=30, look_back=60):
    """Trains a simple GRU model and returns forecast."""
    try:
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create training data
        # Use 80% of data for training
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:train_size, :]

        X_train, y_train = create_dataset(train_data, look_back)
        
        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build the GRU model
        # Using a simple model for quick training
        K.clear_session() # Clear session to avoid model conflicts
        model = Sequential()
        model.add(GRU(50, return_sequences=False, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model (simplified for demo)
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # Generate forecast
        last_sequence = scaled_data[-look_back:]
        forecast = []
        
        current_batch = last_sequence.reshape((1, look_back, 1))

        for _ in range(n_forecast):
            # Get the prediction
            current_pred = model.predict(current_batch, verbose=0)[0]
            forecast.append(current_pred)
            
            # Update the batch
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform the forecast
        forecast = scaler.inverse_transform(forecast)
        return forecast.flatten().tolist()

    except Exception as e:
        print(f"Error in GRU forecast: {e}")
        # Return a list of NaNs as a fallback
        return [np.nan] * n_forecast

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # We will read the index.html file and render it as a string
    # This avoids needing a 'templates' folder
    try:
        with open('index.html', 'r') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "Error: index.html not found. Make sure it's in the same directory as app.py."

@app.route('/api/forecast', methods=['POST'])
def forecast_api():
    """API endpoint to get stock forecasts."""
    try:
        data = request.get_json()
        stock_symbol = data.get('stock')
        
        if not stock_symbol or stock_symbol not in DATA_FILES:
            return jsonify({"error": "Invalid or missing stock symbol."}), 400

        # Load data
        df = load_data(stock_symbol)
        
        # Define number of forecast days
        n_forecast = 30
        
        # Get forecasts
        arima_preds = get_arima_forecast(df['Close'], n_forecast)
        gru_preds = get_gru_forecast(df[['Close']].values, n_forecast)
        
        # Prepare response
        historical_dates = df.index.strftime('%Y-%m-%d').tolist()
        historical_close = df['Close'].tolist()
        
        # Create future dates for the forecast
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)
        future_dates_str = future_dates.strftime('%Y-%m-%d').tolist()

        response = {
            "dates": historical_dates,
            "close": historical_close,
            "future_dates": future_dates_str,
            "arima_forecast": arima_preds,
            "gru_forecast": gru_preds
        }
        
        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

