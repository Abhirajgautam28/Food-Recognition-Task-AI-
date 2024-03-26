import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
from tqdm import tqdm

# Section 1: Dataset Analysis and Preprocessing

# Dataset path
dataset_path = "E:/Applite/Demo work/Source Code/data/food-101/images"

# Preprocessing and Data Augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset and split into train and test
dataset = datasets.ImageFolder(dataset_path, transform=train_transforms)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Section 2: Model Development

# Use a pre-trained ResNet model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))  # Access classes from the original dataset

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Train the model with tqdm progress bar
def train_model_with_progress(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        correct_predictions = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)

            epoch_loss = running_loss / ((batch_idx + 1) * len(inputs))
            epoch_acc = correct_predictions.double() / ((batch_idx + 1) * len(inputs))

            progress_bar.set_postfix(loss=epoch_loss, accuracy=epoch_acc)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


# Evaluate the model with tqdm progress bar
def evaluate_model_with_progress(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            if batch_idx == len(dataloader) - 1:
                progress_bar.set_postfix(accuracy=accuracy_score(y_true, y_pred))

    return y_true, y_pred


# Train the model with tqdm progress
trained_model = train_model_with_progress(model, criterion, optimizer)

# Save the trained model
model_save_path = "E:/Applite/Demo work/Source Code/resnet_model.pth"
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved at: {model_save_path}")

# Evaluate the model on test data with tqdm progress
y_true, y_pred = evaluate_model_with_progress(trained_model, test_loader)

# Classification report
print(classification_report(y_true, y_pred, target_names=dataset.classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(len(dataset.classes)), dataset.classes, rotation=90)  # Use dataset.classes
plt.yticks(np.arange(len(dataset.classes)), dataset.classes)  # Use dataset.classes
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Visualize predictions on sample images
def visualize_predictions(model, loader, num_images=5):
    model.eval()
    classes = dataset.classes  # Use dataset.classes

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(num_images):
                img = inputs[i].permute(1, 2, 0).cpu().numpy()
                true_label = classes[labels[i]]
                pred_label = classes[preds[i]]

                plt.imshow(img)
                plt.title(f'True: {true_label}, Predicted: {pred_label}')
                plt.axis('off')
                plt.show()

                if i == num_images - 1:
                    break

            break


# Visualize predictions on sample images from test dataset
visualize_predictions(trained_model, test_loader)

