import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% Extracting the dataset
df = pd.read_csv('iris.data', header=None)

x = df.iloc[:, :-1].values # features
y, _ = pd.factorize(df.iloc[:, -1]) # target variable

# standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

def to_tensor(data, target):
    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

x_train, y_train = to_tensor(x_train, y_train) # fix typo here
x_test, y_test = to_tensor(x_test, y_test)

# %% RBFN and rbf_kernel definitions
def rbf_kernel(x, centers, beta):
    return torch.exp(-beta * torch.cdist(x, centers) ** 2)

class RBFN(nn.Module):
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))  # centers for the RBF
        self.beta = nn.Parameter(torch.ones(1) * 2.0)  # beta value for the RBF kernel
        self.linear = nn.Linear(num_centers, output_dim)  # Linear layer for classification
    
    def forward(self, x):
        rbf_output = rbf_kernel(x, self.centers, self.beta)  # Apply the RBF kernel
        return self.linear(rbf_output)  # Apply linear transformation after RBF

# %% Model training
num_centers = 10  # Number of RBF centers
model = RBFN(input_dim=4, num_centers=num_centers, output_dim=3)  # Create the model
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Optimizer

num_epochs = 100  # Number of epochs
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero gradients
    outputs = model(x_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# %% Test and evaluate
with torch.no_grad():
    y_pred = model(x_test)  # Forward pass
    accuracy = (torch.argmax(y_pred, axis=1) == y_test).float().mean().item()  # Compute accuracy
    print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy
