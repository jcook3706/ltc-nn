import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchdiffeq import odeint

# Define the Liquid Time-Constant (LTC) neuron
class LTCNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)  # Recurrent weight
        self.U = nn.Linear(input_dim, hidden_dim)   # Input weight
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive time constant
        )
    
    def forward(self, t, state, x):
        tau = self.tau_net(torch.cat([state, x], dim=-1)) + 1e-2  # Avoid division by zero
        dh_dt = (self.W(state) + self.U(x) + self.b - state) / tau
        return dh_dt

# Define the LTC-based RNN model
class LTCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ltc_cell = LTCNeuron(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, hidden_dim, device=x.device)  # Initial hidden state
        t_span = torch.linspace(0, seq_len, seq_len, device=x.device)
        
        # Solve the ODE for each time step
        h_seq = odeint(self.ltc_cell, h, t_span, args=(x,))
        h_final = h_seq[-1]  # Take last time step output
        return self.output_layer(h_final)

# Define hyperparameters
input_dim = 28  # Each row of an MNIST image (28x28)
hidden_dim = 128
output_dim = 10  # 10 classes (digits 0-9)
batch_size = 64
epochs = 10
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Sequential MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze().transpose(0, 1))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss, and optimizer
model = LTCNN(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/len(train_dataset):.4f}")

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(dataloader.dataset)

# Evaluate on test data
test_accuracy = evaluate(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")
