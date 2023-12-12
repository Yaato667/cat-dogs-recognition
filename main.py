import os
import torch
from torch import nn
from torch.utils.data import TensorDataset


# On doit ensuite dire à PyTorch quel matériel cibler sur la machine pour effectuer nos opérations
device = (
    "cuda" if torch.cuda.is_available()  # GPU nvidia
    else "mps" if torch.backends.mps.is_available()  # Certains systèmes Mac
    else "cpu"  # Le reste
)

print(f"{device} is being used")

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()  # Appel du constructeur de sa classe mère nn.Module
        self.archiNN = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.archiNN(x)
        return out

# Une fois la classe Perceptron faite, je l'instancie en disant à PyTorch d'utiliser le matériel détecté
monPerceptron = Perceptron().to(device)

# Remplacez les chaînes vides par les répertoires réels de vos données d'entraînement et de test
train_data_dir_cat = 'C:/Users/tomla/Downloads/training_set/training_set/cats'
train_data_dir_dog = 'C:/Users/tomla/Downloads/training_set/training_set/dogs'
test_data_dir_cat = 'C:/Users/tomla/Downloads/test_set/test_set/cats'
test_data_dir_dog = 'C:/Users/tomla/Downloads/test_set/test_set/dogs'

# Chargez vos données d'entraînement et de test ici, par exemple avec torchvision.datasets.ImageFolder
# (Vous devrez installer torchvision si vous ne l'avez pas déjà fait)
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor()])

train_dataset_cat = datasets.ImageFolder(root=train_data_dir_cat, transform=transform)
train_dataset_dog = datasets.ImageFolder(root=train_data_dir_dog, transform=transform)
test_dataset_cat = datasets.ImageFolder(root=test_data_dir_cat, transform=transform)
test_dataset_dog = datasets.ImageFolder(root=test_data_dir_dog, transform=transform)

# Concaténez les ensembles d'entraînement et de test pour les chats et les chiens
train_dataset = torch.utils.data.ConcatDataset([train_dataset_cat, train_dataset_dog])
test_dataset = torch.utils.data.ConcatDataset([test_dataset_cat, test_dataset_dog])

# Créez les TensorDataset à partir des ensembles d'entraînement et de test
tensor_train_set = TensorDataset(*zip(*train_dataset))
tensor_test_set = TensorDataset(*zip(*test_dataset))
