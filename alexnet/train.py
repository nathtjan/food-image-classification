import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from . import AlexNet


class ImageDFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "class_id"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def densify_labels(df: pd.DataFrame, label_col="class_id"):
    """Convert sparse class ids into dense 0..N-1"""
    unique_labels = sorted(df[label_col].unique())
    label2dense = {orig: i for i, orig in enumerate(unique_labels)}
    dense2label = {i: orig for i, orig in enumerate(unique_labels)}

    df = df.copy()
    df[label_col] = df[label_col].map(label2dense)
    return df, label2dense, dense2label


def train_alexnet_from_df(train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                          batch_size=32, epochs=10, lr=0.001, num_workers=2,
                          init_state_dict=None):

    # === Preprocess: make labels dense ===
    train_df, label2dense, dense2label = densify_labels(train_df, "class_id")
    num_classes = len(label2dense)

    if val_df is not None:
        val_df = val_df.copy()
        val_df["class_id"] = val_df["class_id"].map(label2dense)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | num_classes={num_classes}")
    print(f"Label mapping: {label2dense}")

    # === Transforms ===
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # === Datasets & Loaders ===
    train_dataset = ImageDFDataset(train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = None
    if val_df is not None:
        val_dataset = ImageDFDataset(val_df, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    # === Model ===
    model = AlexNet(num_classes=num_classes)
    if init_state_dict:
        model.load_state_dict(init_state_dict)
    model = model.to(device)

    # === Loss & Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        log = f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"

        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total
            log += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"

        print(log)

    return model, label2dense, dense2label
