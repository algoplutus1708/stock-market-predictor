from flask import Flask, render_template, jsonify, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Load the datasets
google = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
microsoft = pd.read_csv('MSFT_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
ibm = pd.read_csv('IBM_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
amazon = pd.read_csv('AMZN_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

datasets = {
    'GOOGL': google,
    'MSFT': microsoft,
    'IBM': ibm,
    'AMZN': amazon
}

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
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model.load_state_dict(torch.load('gru_model.pth'))
model.eval()

def get_arima_forecast(data):
    model_arima = ARIMA(data['Close'], order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    forecast = model_arima_fit.forecast(steps=30)
    return forecast

def get_gru_forecast(data):
    # Prepare data for GRU model
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    close_price = data['Close'].values.reshape(-1, 1)
    test_data = scaler.fit_transform(close_price)
    test_data = torch.from_numpy(test_data).type(torch.Tensor)
    test_data = test_data.unsqueeze(0)
    # Get prediction
    prediction = model(test_data)
    prediction = scaler.inverse_transform(prediction.detach().numpy())
    return prediction.flatten()


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
    
    # ARIMA forecast
    arima_forecast = get_arima_forecast(data)
    
    # GRU forecast
    gru_forecast = get_gru_forecast(data)

    return jsonify({
        'dates': data.index.strftime('%Y-%m-%d').tolist(),
        'close': data['Close'].tolist(),
        'arima_forecast': arima_forecast.tolist(),
        'gru_forecast': gru_forecast.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)