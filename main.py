# On utilise les imports suivants pour inclure PyTorch dans notre projet
import os
import torch
from torch import nn

# On doit ensuite dire à PyTorch quel matériel cibler sur la machine pour effectuer nos opération
device = (
    "cuda" if torch.cuda.is_available() # GPU nvidia
    else "mps" if torch.backends.mps.is_available() # ertains systèmes Mac
    else "cpu" # Le reste
)

print(f"{device} is being used")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt



# Define the Perceptron class
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.archiNN = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1),  # Adjust the input size to match the flattened image size
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.archiNN(x)
        return out

# Instantiate the Perceptron and move it to the specified device
monPerceptron = Perceptron().to(device)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load the training data
train_dataset = ImageFolder(root='/content/train', transform=transform)

# Load the testing data
test_dataset = ImageFolder(root='/content/test', transform=transform)

# Define data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(monPerceptron.parameters(), lr=0.1)

# Lists to store training information
train_losses = []
test_accuracies = []

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        batch_size = inputs.size(0)  # Get the current batch size
        inputs, labels = inputs.view(batch_size, -1).to(device), labels.float().view(batch_size, 1).to(device)

        # Forward pass
        outputs = monPerceptron(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save training loss
    train_losses.append(loss.item())

    # Testing loop
    monPerceptron.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            batch_size = inputs.size(0)  # Get the current batch size
            inputs, labels = inputs.view(batch_size, -1).to(device), labels.float().view(batch_size, 1).to(device)
            outputs = monPerceptron(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        test_accuracies.append(accuracy)

    monPerceptron.train()  # Set the model back to training mode

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Test Accuracy: {accuracy * 100:.2f}%')

# Plot training loss and test accuracy curves
plt.figure(figsize=(12, 4))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

# Plot Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()
