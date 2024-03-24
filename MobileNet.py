import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import mobilenet_v2


# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.num_classes = len(self.classes)
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_images()

    def load_images(self):
        for cls in self.classes:
            img_names = np.random.choice(os.listdir(os.path.join(self.data_dir, cls)),
                                         num_images_to_load // self.num_classes, replace=False)
            for img_name in img_names:
                img_path = os.path.join(self.data_dir, cls, img_name)
                self.images.append(img_path)
                self.labels.append(cls)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "E:\\Applite\\Demo work\\Source Code\\data\\food-101\\images"
num_images_to_load = 101000  # Define or calculate the actual number of images to load
dataset = CustomDataset(data_dir, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders with adjusted batch size
batch_size = 64  # Adjust the batch size for faster training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained MobileNetV2 model with reduced width
print("Loading pre-trained MobileNetV2 model...")
model = mobilenet_v2(pretrained=True) # Use MobileNetV2 with reduced width (50% of original width)

# Modify the last fully connected layer of the classifier to match the number of classes in your dataset
num_classes = len(dataset.classes)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Map class names to numerical labels
class_to_label = {cls: i for i, cls in enumerate(dataset.classes)}

# Train the model
print("Training the model...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):  # Iterate through train_loader
        # Convert labels to numerical representations
        labels = torch.tensor([class_to_label[label] for label in labels]).to(device)
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print batch progress
        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                  f"Train Loss: {loss.item():.4f}")

    # Calculate and print epoch statistics
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = torch.tensor([class_to_label[label] for label in labels]).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_loss /= len(test_loader)
    val_acc = 100. * correct / total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print("Model training completed.")

    # Save the model
    torch.save(model.state_dict(), 'mobilenet_model.pth')
    print("Model saved.")

    # Print CUDA information
    print(f"CUDA is available: {torch.cuda.is_available()}")

    # Load saved model
    print("Loading saved model...")
    model = torchvision.models.mobilenet_v2(width_mult=0.5, pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load('mobilenet_model.pth'))
    model.eval()
    print("Model loaded.")

    # Evaluate the trained model
    print("Evaluating the trained model...")
    test_losses = []
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Corrected line
            test_losses.append(loss.item())
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss = np.mean(test_losses)
    test_accuracy = 100. * test_correct / test_total
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Visualize model's predictions on sample images
    print("Visualizing model's predictions on sample images...")
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)  # Get predictions from the model

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        true_label = dataset.classes[labels[i].item()]
        pred_label = dataset.classes[predicted[i].item()]
        ax.set_title(f'True: {true_label}\nPred: {pred_label}')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("Evaluation and testing completed.")

