import os
import random
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm  # Import tqdm for progress bar
from RSU import REBNCONV, RSU7, RSU6, RSU5, RSU4, RSU4F

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define label conversion function for font labels
font_labels = {
    "395-CAI978": 0, "625-CAI978": 1, "525-CAI978": 2, "600-CAI978": 3, 
    "470-CAI978": 4, "396-CAI978": 5, "868-CAI978": 6, "330-CAI978": 7, 
    "728-CAI978": 8, "656-CAI978": 9, "524-CAI978": 10, "888-CAI978": 11, 
    "649-CAI978": 12, "593-CAI978": 13, "949-CAI978": 14, "328-CAI978": 15, 
    "861-CAI978": 16, "535-CAI978": 17, "662-CAI978": 18, "657-CAI978": 19, 
    "474-CAI978": 20, "697-CAI978": 21, "563-CAI978": 22, "431-CAI978": 23, 
    "357-CAI978": 24, "471-CAI978": 25, "602-CAI978": 26, "552-CAI978": 27, 
    "612-CAI978": 28, "345-CAI978": 29, "876-CAI978": 30, "346-CAI978": 31, 
    "560-CAI978": 32, "545-CAI978": 33, "361-CAI978": 34, "717-CAI978": 35, 
    "301-CAI978": 36, "389-CAI978": 37, "481-CAI978": 38, "629-CAI978": 39, 
    "921-CAI978": 40, "321-CAI978": 41, "622-CAI978": 42, "713-CAI978": 43, 
    "383-CAI978": 44, "604-CAI978": 45, "464-CAI978": 46, "451-CAI978": 47, 
    "575-CAI978": 48, "309-CAI978": 49, "485-CAI978": 50, "925-CAI978": 51, 
    "459-cai978": 52, "391-CAI978": 53, "708-CAI978": 54, "422-CAI978": 55, 
    "680-CAI978": 56, "571-CAI978": 57, "581-CAI978": 58, "349-CAI978": 59, 
    "303-CAI978": 60, "513-CAI978": 61, "627-CAI978": 62, "635-CAI978": 63, 
    "356-CAI978": 64, "502-CAI978": 65, "912-CAI978": 66, "527-CAI978": 67, 
    "902-CAI978": 68, "447-CAI978": 69, "477-CAI978": 70, 
    "510-CAI978": 71, "359-CAI978": 72, "646-CAI978": 73, "308-CAI978": 74, 
    "632-CAI978": 75, "492-CAI978": 76
}

def conv_label(label):
    # Convert font label to its corresponding index
    return font_labels.get(label, -1)

def pil_image(img_path):
    # Load and resize image to 105x105
    pil_im = Image.open(img_path)
    pil_im = pil_im.resize((105, 105))
    return pil_im

class FontDataset(Dataset):
    # Custom dataset for loading font images and their labels
    def __init__(self, imagePaths, labels, transform=None):
        self.imagePaths = imagePaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # Load an image and its label
        img_path = self.imagePaths[idx]
        label = self.labels[idx]
        image = pil_image(img_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label

# Data augmentation and normalization for training and testing
train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Load image paths and labels
data_path = "./font_patch/"
imagePaths = [
    os.path.join(dp, f)
    for dp, dn, filenames in os.walk(data_path)
    for f in filenames
    if f.endswith((".png", ".jpg", ".jpeg"))
]
random.seed(42)
random.shuffle(imagePaths)

# Generate labels from directory names
labels = [conv_label(os.path.basename(os.path.dirname(p))) for p in imagePaths]

# Split data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(
    imagePaths, labels, test_size=0.25, random_state=42
)

# Create dataset objects for training and testing
train_dataset = FontDataset(trainX, trainY, transform=train_transform)
test_dataset = FontDataset(testX, testY, transform=test_transform)

# Create DataLoader objects for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

class U2NET_Classification(nn.Module):
    # Define U²-Net model with classification head
    def __init__(self, in_ch=3, out_ch=78):  # Adjust out_ch to match the number of classes
        super(U2NET_Classification, self).__init__()

        # Define the stages of the U²-Net
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        self.fc = nn.Linear(512 * 7 * 7, out_ch)  # Fully connected layer to output class scores

    def forward(self, x):
        # Forward pass through the network
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx5)

        hx6_flat = torch.flatten(hx6, 1)
        out = self.fc(hx6_flat)

        return out

model = U2NET_Classification(in_ch=3, out_ch=78).to(device)

# Initialize loss function, optimizer, and learning rate scheduler
learning_rate = 0.0001
weight_decay = 1e-4  # Optional weight decay for regularization

criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam optimizer
scheduler = optim.lr
