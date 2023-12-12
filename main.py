import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# Assuming 'device' is defined and contains the PyTorch device (e.g., cuda or cpu)
# You can define it as follows:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Redéfinir la classe du modèle en utilisant une architecture CNN (ResNet18)
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(ImageClassifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet18(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model and move it to the specified device
image_classifier = ImageClassifier().to(device)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Load the training data
train_dataset = ImageFolder(root='/content/train', transform=transform)

# Load the testing data
test_dataset = ImageFolder(root='/content/test', transform=transform)

# Define data loaders with num_workers
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define loss function and optimizer (use Adam optimizer)
criterion = nn.BCELoss()
optimizer = optim.Adam(image_classifier.parameters(), lr=0.001)

# Function to train the model
def train(train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().view(-1, 1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='binary')

    return running_loss / len(train_loader), accuracy, precision

# Lists to store training information
train_losses = []
train_accuracies = []
train_precisions = []

test_losses = []
test_accuracies = []
test_precisions = []

num_epochs = 10
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}\n-------------------------------")

    # Entraînement
    train_loss, train_accuracy, train_precision = train(train_loader, image_classifier, criterion, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    print(f'Training Loss: {train_loss}, Accuracy: {train_accuracy}, Precision: {train_precision}')

    # Validation
    test_loss, test_accuracy, test_precision = test(test_loader, image_classifier, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    test_precisions.append(test_precision)
    print(f'Testing Loss: {test_loss}, Accuracy: {test_accuracy}, Precision: {test_precision}')

# Fonction pour tracer les métriques
def plot_metrics(train_values, test_values, metric_name):
    plt.plot(train_values, label=f'Training {metric_name}')
    plt.plot(test_values, label=f'Testing {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# Tracer les métriques
plot_metrics(train_losses, test_losses, 'Loss')
plot_metrics(train_accuracies, test_accuracies, 'Accuracy')
plot_metrics(train_precisions, test_precisions, 'Precision')
