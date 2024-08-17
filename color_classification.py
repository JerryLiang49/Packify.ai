import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from diffusers.optimization import get_scheduler
from tqdm import tqdm  # For progress bar

def hex_to_rgb(hex_str):
    # Convert HEX color string to normalized RGB values
    hex_str = hex_str.lstrip("#")
    return [int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]

class ColorTextDataset(Dataset):
    # Custom dataset for loading images and RGB color labels
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image and its RGB color label
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        hex_color = img_path.split("_")[-1].split(".")[0]
        label = hex_to_rgb(hex_color)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)

class ColorRegressorCNN(nn.Module):
    # Convolutional Neural Network for RGB color regression
    def __init__(self):
        super(ColorRegressorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Second convolutional layer
        self.conv3 = nn.Conv2d(32, 64, 3, 1)  # Third convolutional layer
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 3)  # Output layer for RGB values

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Ensure output is between 0 and 1
        return x

# Hyperparameters
batch_size = 512
learning_rate = 1e-3
num_epochs = 50
best_val_loss = float("inf")  # Initialize best validation loss

# Data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,)),  # Normalize images
])

# Load dataset
dataset = ColorTextDataset(root_dir="./models/color_patch/", transform=transform)
print(f"Dataset size: {len(dataset)}")

# Split dataset into training and validation sets
train_indices, val_indices = train_test_split(
    list(range(len(dataset))), test_size=0.01, random_state=42
)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorRegressorCNN().to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Learning rate scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=num_epochs * len(train_loader),
)

def calculate_error(predictions, labels):
    # Calculate custom error metric
    diff = torch.abs(predictions - labels)
    avg_diff = torch.sum(diff, dim=1) / 3
    return torch.sum(avg_diff).item() / len(labels)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.item()

            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
            pbar.update(1)

    # Validation
    model.eval()
    val_loss = 0.0
    custom_accuracy = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            custom_accuracy += calculate_error(outputs, labels) * labels.size(0)

    val_loss /= len(val_loader)
    custom_accuracy /= len(val_loader.dataset)

    print(f"Validation Loss: {val_loss:.4f}, Error: {100*custom_accuracy:.4f}%")

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_color_model.pth")

print("Training complete.")

# Save the final model
torch.save(model.state_dict(), "last_color_model.pth")

# Load the best model for inference
model.load_state_dict(torch.load("best_color_model.pth"))
