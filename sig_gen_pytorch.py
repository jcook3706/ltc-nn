import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ------------------------- Signal Generator -------------------------
def generate_signal(length, dims, num_components=3, trend_range=(-0.1, 0.1)):
    t = np.linspace(0, 10, length)
    signals = []
    for d in range(dims):
        signal = np.zeros_like(t)
        freqs = np.random.uniform(1, 20, size=num_components)
        weights = np.random.uniform(0.1, 2.0, size=num_components)
        for freq, weight in zip(freqs, weights):
            signal += weight * np.sin(2 * np.pi * freq * t)
        trend_slope = np.random.uniform(trend_range[0], trend_range[1])
        signal += trend_slope * t  # linear trend
        signals.append(signal)
    return np.stack(signals, axis=-1)

# ------------------------- LSTM Model -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=None):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size or input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# ------------------------- Training & Evaluation -------------------------
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            total_loss += criterion(output, y).item()
    return total_loss / len(dataloader)

# ------------------------- Main -------------------------
if __name__ == '__main__':
    # Parameters
    dims = 8
    signal_length = 1000
    sequence_length = 50
    batch_size = 32
    epochs = 20

    # Generate signal and save
    signal = generate_signal(signal_length, dims)
    np.save('generated_signal.npy', signal)

    # Prepare data for LSTM
    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            x = data[i:i+seq_len]
            y = data[i+1:i+seq_len+1]
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

    x, y = create_sequences(signal, sequence_length)
    dataset = TensorDataset(x, y)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model setup
    model = LSTMModel(input_size=dims, output_size=dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer)
        train_loss = evaluate_model(model, train_loader, criterion)
        test_loss = evaluate_model(model, test_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

    # Evaluate on test
    model.eval()
    x_test, y_test = next(iter(test_loader))
    with torch.no_grad():
        y_pred = model(x_test)

    # Plot predictions
    for i in range(0, dims, 4):
        plt.figure(figsize=(12, 10))
        for j in range(4):
            if i + j >= dims:
                break
            plt.subplot(2, 2, j+1)
            plt.plot(y_test[0,:,i+j], label='True')
            plt.plot(y_pred[0,:,i+j], label='Predicted', linestyle='--')
            plt.title(f'Dimension {i+j+1}')
            plt.legend()
        plt.tight_layout()
        plt.show()
