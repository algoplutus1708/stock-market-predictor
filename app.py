import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print(" --- Script starting --- ")

# --- Step 1: Check for essential files ---
required_files = [
    'gru_model.pth',
    'GOOGL_2006-01-01_to_2018-01-01.csv',
    'MSFT_2006-01-01_to_2018-01-01.csv',
    'IBM_2006-01-01_to_2018-01-01.csv',
    'AMZN_2006-01-01_to_2018-01-01.csv'
]

for filename in required_files:
    if not os.path.exists(filename):
        print(f"!!! FATAL ERROR: Required file not found: {filename}")
        print("!!! Please make sure all required files are in the same folder as app.py.")
        exit() # Stop the script if a file is missing
    else:
        print(f"File found: {filename}")


print("\n --- Initializing Flask App --- ")
app = Flask(__name__)

# Load the datasets
print("Loading datasets...")
google = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
microsoft = pd.read_csv('MSFT_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
ibm = pd.read_csv('IBM_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
amazon = pd.read_csv('AMZN_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
datasets = {'GOOGL': google, 'MSFT': microsoft, 'IBM': ibm, 'AMZN': amazon}
print("Datasets loaded successfully.")

# Define the GRU model
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Load the trained GRU model
print("Loading trained GRU model...")
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model.load_state_dict(torch.load('gru_model.pth'))
model.eval()
print("Model loaded successfully.")

def get_arima_forecast(data, forecast_steps=30):
    # ... (rest of the code is the same)
    model_arima = ARIMA(data['Close'], order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    forecast = model_arima_fit.forecast(steps=forecast_steps)
    return forecast

def get_gru_forecast(data, forecast_steps=30, train_window=12):
    close_price = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(close_price)
    
    test_inputs = data_normalized[-train_window:]
    test_inputs = torch.FloatTensor(test_inputs).view(1, train_window, 1)

    future_predictions = []
    for _ in range(forecast_steps):
        with torch.no_grad():
            prediction = model(test_inputs)
        
        future_predictions.append(prediction.item())
        new_sequence = test_inputs.numpy().flatten()
        new_sequence = np.append(new_sequence[1:], prediction.item())
        test_inputs = torch.FloatTensor(new_sequence).view(1, train_window, 1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stocks')
def get_stocks():
    return jsonify(list(datasets.keys()))

@app.route('/api/forecast', methods=['POST'])
def forecast():
    stock = request.json['stock']
    data = datasets[stock]
    forecast_steps = 30
    
    arima_forecast = get_arima_forecast(data, forecast_steps)
    gru_forecast = get_gru_forecast(data, forecast_steps)

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps).strftime('%Y-%m-%d').tolist()

    return jsonify({
        'dates': data.index.strftime('%Y-%m-%d').tolist(),
        'close': data['Close'].tolist(),
        'future_dates': future_dates,
        'arima_forecast': arima_forecast.tolist(),
        'gru_forecast': gru_forecast.tolist()
    })

print("\n --- Starting server --- ")
if __name__ == '__main__':
    print("Executing app.run()...")
    app.run(debug=True)
else:
    print("Warning: app.run() was not called because __name__ is not '__main__'.")