import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print("Device:",device)

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

print(data)

X = data.drop('quality', axis=1)
y = data['quality'] - 3

X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameters
num_features = X_train_scaled.shape[1]
num_classes = 6
epochs = 100

class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim_embedding=64, num_heads=4, num_layers=4):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, num_classes)

    def forward(self, x):
        x = x.to(device)
        x = self.embedding(x)
        x = x.unsqueeze(dim=1)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x


model = TabTransformer(num_features, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)

losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_tensor).to(device)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

plt.figure(figsize=(10,6))
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid("True")
plt.show()

model.eval()
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

with torch.no_grad():
    predictions = model(X_test_tensor).to(device)
    _, predicted_classes = torch.max(predictions, 1)
    accuracy = (predicted_classes == y_test_tensor).float().mean()
    print(f'Test accuracy: {accuracy.item()}')

