import os
import shutil
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --------------------
# パラメータ設定
# --------------------
base_dir = "camera_dataset"
original_dir = os.path.join(base_dir, "train")
train_dir = os.path.join(base_dir, "train_split")
val_dir = os.path.join(base_dir, "val")
batch_size = 32
num_classes = 6
num_epochs = 10
learning_rate = 0.0005#0.0005にて損失曲線が収束した。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# データを train/val に 8:2 自動分割
# --------------------
def prepare_split_data():
    print("データを 8:2 に分割中...")
    for label in range(1, 7):
        label_str = str(label)
        src_dir = os.path.join(original_dir, label_str)
        train_label_dir = os.path.join(train_dir, label_str)
        val_label_dir = os.path.join(val_dir, label_str)

        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        all_images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png'))]
        random.shuffle(all_images)

        split_index = int(len(all_images) * 0.8)
        train_images = all_images[:split_index]
        val_images = all_images[split_index:]

        for img in train_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(train_label_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(val_label_dir, img))

    print("分割完了")

# --------------------
# 前処理とデータローダー
# --------------------
def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# --------------------
# モデル定義と学習
# --------------------
def train_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_dataloaders()
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_loss)

        # 検証フェーズ/lerobot/outputs/train/pick_saikoro1/checkpoints/last/pretrained_model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        val_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "dice_classifier.pth")
    print("モデル保存完了 → dice_classifier.pth")

    # 学習曲線
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()


# --------------------
# 実行
# --------------------
if __name__ == "__main__":
    prepare_split_data()
    train_model()
