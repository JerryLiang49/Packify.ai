import os
import random
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from u2net import data_loader, u2net
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
width, height = 320, 320
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "model_124"
epochs = 100
batch_size = 8
learning_rate = 1e-4
data_path = "/home/baoxiaohe/jerry/remove_background/"
checkpoint_dir = "checkpoints"
output_dir = "outputs"

# Ensure directories exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Initialize model
net = u2net.U2NET(3, 1).to(device=device, dtype=torch.float32)
net.train()

# Normalization function
def norm_pred(d):
    ma, mi = torch.max(d), torch.min(d)
    return (d - mi) / (ma - mi)

# Preprocess function
def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[:2])

    if len(label_3.shape) == 3:
        label = label_3[:, :, 0]

    if len(image.shape) == 3 and len(label.shape) == 2:
        label = label[:, :, np.newaxis]

    transform = transforms.Compose(
        [data_loader.RescaleT(320), data_loader.ToTensorLab(flag=0)]
    )
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})
    return sample

# Image transformations
transform = transforms.Compose(
    [transforms.Resize((height, width)), transforms.ToTensor()]
)

_imgTransform = transforms.Compose(
    [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Check and delete truncated images
def delete_truncated_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the file is not truncated
            except (UnidentifiedImageError, IOError):
                print(f"Deleting truncated image: {file_path}")
                os.remove(file_path)

# Dataset and DataLoader
class RemoverDataset(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        try:
            origin = Image.open(
                os.path.join(data_path, "output2", name)
            ).convert("RGB")
            _dot_index = name.rfind(".")
            removed = Image.open(
                os.path.join(data_path, "result2", f"{name[:_dot_index]}.png")
            )
            mask = removed.convert("L")  # Ensure mask is grayscale
            mask = transform(mask)
            sample = _imgTransform(origin)
            return {"mask": mask, "origin": sample}
        except Exception as e:
            print(f"Error loading data for {name}: {e}")
            return None

# Delete truncated images in the output folder
delete_truncated_images(os.path.join(data_path, "output2"))

# Load dataset
files = os.listdir(os.path.join(data_path, "output2"))
names = [item for item in files if "." in item]
dataset = RemoverDataset(names=names)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"Number of batches: {len(train_loader)}")

# Training components
criterion = torch.nn.MSELoss()
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=epochs * len(train_loader),
)

# Naive Cutout Function
def naive_cutout(img, mask):
    empty = Image.new("RGBA", img.size, (0, 0, 0, 0))
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout

# Alpha Matting Cutout Function
def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size
    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)
    img = np.asarray(img)
    mask = np.asarray(mask)
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold
    structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int64) if erode_structure_size > 0 else None
    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0
    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)
    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)
    return cutout

# Evaluation function
def evaluate(epoch, net, use_alpha_matting=False):
    try:
        random_element = random.choice(names)
        origin = Image.open(
            os.path.join(data_path, "output2", f"{random_element}")
        ).convert("RGB")
        sample = _imgTransform(origin)
        sample_image = sample.to(device, dtype=torch.float32).unsqueeze(0)
        net.eval()
        with torch.no_grad():
            d1, *_ = net(sample_image)
            pred = d1[:, 0, :, :]
            predict = norm_pred(pred)
            predict_image = transforms.ToPILImage()(predict.cpu().squeeze(0)).convert("L")
            cutout = alpha_matting_cutout(origin, predict_image, 240, 10, 10, 1000) if use_alpha_matting else naive_cutout(origin, predict_image)
            cutout.save(os.path.join(output_dir, f"{epoch}.png"))
            origin.save(os.path.join(output_dir, f"{epoch}.jpg"))
        net.train()
    except Exception as e:
        print(f"Error during evaluation at epoch {epoch}: {e}")

min_loss = float('inf')

# Training loop
for epoch in range(epochs):
    net.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    total_loss = 0
    for item in train_loader:
        if item is None:
            continue
        origin = item["origin"].to(device, dtype=torch.float32)
        mask = item["mask"].to(device, dtype=torch.float32).squeeze(dim=1)
        optimizer.zero_grad()
        d1, *_ = net(origin)
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)
        loss = criterion(predict.float(), mask.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
        pbar.update(1)
    pbar.close()
    
    # Save model checkpoints
    if total_loss < min_loss:
        min_loss = total_loss
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, "model_best.pt"))
    
    if (epoch + 1) % 5 == 0:
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}.pt"))
    
    evaluate(epoch, net, use_alpha_matting=False)  # Change to True to use alpha matting
