import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define the EXACT SAME model class as in app.py
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

# Load and prepare data
print("Loading data for training...")
data = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
close_price = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(close_price)
data_normalized = torch.FloatTensor(data_normalized).view(-1)

# Define a function to create sequences
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_window = 12
train_inout_seq = create_inout_sequences(data_normalized, train_window)

# Instantiate the model and define loss/optimizer
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
# --- KEY CHANGE: Lower learning rate for more stable training ---
optimiser = torch.optim.Adam(model.parameters(), lr=0.005)

# --- KEY CHANGE: Increased epochs for more thorough training ---
epochs = 150 
print(f"Starting training for {epochs} epochs...")

# Train the model
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimiser.zero_grad()
        y_pred = model(seq.view(-1, train_window, 1))
        single_loss = criterion(y_pred, labels.view(-1,1))
        single_loss.backward()
        optimiser.step()

    if (i+1) % 25 == 0: # Print progress every 25 epochs
        print(f'epoch: {i+1:3} loss: {single_loss.item():10.8f}')

# Save the trained model
torch.save(model.state_dict(), 'gru_model.pth')
print("\nNew, smarter GRU model has been trained and saved as gru_model.pth")