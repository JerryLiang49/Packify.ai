import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 12
num_classes = 10
epochs = 20
learn_rate = 1e-2

# Dataset and DataLoader
train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

train_ds = torchvision.datasets.MNIST('data', train=True, transform=train_transform, download=True)
test_ds = torchvision.datasets.MNIST('data', train=False, transform=test_transform, download=True)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# Define the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # First convolution layer
        self.pool1 = nn.MaxPool2d(2)                  # Pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # Second convolution layer
        self.pool2 = nn.MaxPool2d(2) 
        self.fc1 = nn.Linear(64 * 5 * 5, 64)          # Fully connected layer
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = Model().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=learn_rate)

# Training function
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    accuracy = correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

# Testing function
def test(dataloader, model, loss_fn):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, target)
            total_loss += loss.item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    
    accuracy = correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

# Training and evaluation loop
train_loss, train_acc = [], []
test_loss, test_acc = [], []

for epoch in range(epochs):
    train_epoch_acc, train_epoch_loss = train(train_dl, model, loss_fn, opt)
    test_epoch_acc, test_epoch_loss = test(test_dl, model, loss_fn)
    
    train_acc.append(train_epoch_acc)
    train_loss.append(train_epoch_loss)
    test_acc.append(test_epoch_acc)
    test_loss.append(test_epoch_loss)
    
    print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_epoch_acc*100:.1f}%, Train Loss: {train_epoch_loss:.3f}, Test Acc: {test_epoch_acc*100:.1f}%, Test Loss: {test_epoch_loss:.3f}")

# Save model
torch.save(model.state_dict(), './model.pt')
print('Model saved.')

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_acc, label='Training Accuracy')
plt.plot(range(epochs), test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_loss, label='Training Loss')
plt.plot(range(epochs), test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')

plt.tight_layout()
plt.savefig("result.png")
plt.show()
 