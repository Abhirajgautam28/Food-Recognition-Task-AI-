import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

# Section 1: Dataset Analysis and Preprocessing

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load downloaded dataset
print("Loading dataset...")
train_set = torchvision.datasets.Food101(root='./data', split='train', transform=transform)
test_set = torchvision.datasets.Food101(root='./data', split='test', transform=transform)

print("Dataset loaded.")

# Split dataset into training and testing sets
train_size = int(0.8 * len(train_set))
test_size = len(train_set) - train_size
train_set, val_set = torch.utils.data.random_split(train_set, [train_size, test_size])

# Create data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

print("Data loaders created.")

# Section 2: Model Development

# Define the CNN model
class FoodClassifier(nn.Module):
    def __init__(self):
        super(FoodClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 101)  # Assuming 101 classes in Food-101 dataset
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = FoodClassifier()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Print progress for each batch
            print(f"\rBatch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}", end='')
        end_time = time.time()
        # Print training loss after each epoch
        print(f"\nEpoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Time: {end_time - start_time:.2f} seconds")
        # Evaluate the model on validation set after each epoch
        evaluate_model(model, val_loader)


# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))


# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model on the test set
evaluate_model(model, test_loader)

# Section 4: Optimization and Deployment

# Add data augmentation for optimization
transform_augmented = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use augmented data for training
print("Loading augmented data...")
train_set_augmented = torchvision.datasets.Food101(root='./data', split='train', transform=transform_augmented)
print("Augmented data loaded.")
train_loader_augmented = torch.utils.data.DataLoader(train_set_augmented, batch_size=batch_size, shuffle=True)

# Retrain the model with augmented data
model_augmented = FoodClassifier()
optimizer_augmented = optim.Adam(model_augmented.parameters(), lr=0.001)
train_model(model_augmented, train_loader_augmented, val_loader, criterion, optimizer_augmented, num_epochs=10)

# Save the trained model
torch.save(model_augmented.state_dict(), 'food_classifier.pth')

print("Training completed and model saved.")
