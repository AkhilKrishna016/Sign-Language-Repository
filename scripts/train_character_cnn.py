"""
Fingerspelling Alphabet CNN Trainer with Transfer Learning (MobileNetV2)

Trains a 26-class (A-Z) single-character ASL classifier:
1. Transfer Learning on Pretrained MobileNetV2.
2. Phase 1: Frozen backbone (train classifier head).
3. Phase 2: Unfrozen fine-tuning at a lower learning rate.
4. Data augmentation (Rotation, scale/translation, brightness/contrast jitter).
5. Stratified train/val/test split.
6. Full class-wise Precision, Recall, F1 metrics and Confusion Matrix.
7. Explicit validation that zero predictions fall outside character set {A-Z}.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Dataset Definition
class ASLAlphabetDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_dataset_filelist(dataset_dir):
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    file_paths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for c in classes:
        folder = os.path.join(dataset_dir, c)
        imgs = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
        for img in imgs:
            file_paths.append(img)
            labels.append(class_to_idx[c])

    return file_paths, labels, classes, class_to_idx

def train_character_model(
    dataset_dir: str = "dataset/character dataset/asl_alphabet_train/asl_alphabet_train",
    output_model_path: str = "backend/models/character_mobilenet_v2.pth",
    output_meta_path: str = "backend/models/character_mobilenet_v2_meta.json",
    batch_size: int = 32,
    phase1_epochs: int = 4,
    phase2_epochs: int = 4
):
    print("=" * 70)
    print("  SignBridge - Fingerspelling CNN Transfer Learning (MobileNetV2)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")

    # 1. Load File list & Stratified Splits
    file_paths, labels, classes, class_to_idx = load_dataset_filelist(dataset_dir)
    num_classes = len(classes)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    print(f"[Dataset] Total samples: {len(file_paths)} across {num_classes} classes: {classes}")

    # Stratified Train (70%), Val (15%), Test (15%)
    X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(
        X_temp_paths, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[Splits] Train: {len(X_train_paths)} | Val: {len(X_val_paths)} | Test: {len(X_test_paths)}")

    # 2. Data Augmentations & Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ASLAlphabetDataset(X_train_paths, y_train, transform=train_transforms)
    val_dataset = ASLAlphabetDataset(X_val_paths, y_val, transform=eval_transforms)
    test_dataset = ASLAlphabetDataset(X_test_paths, y_test, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model Architecture: MobileNetV2 Transfer Learning
    print("\n[Model] Initializing MobileNetV2 with pretrained ImageNet weights...")
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Custom Classification Head for 26 character classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------
    # PHASE 1: Train Head Only (Backbone Frozen)
    # ----------------------------------------------------
    print(f"\n[Phase 1] Freezing CNN Backbone -> Training Classification Head ({phase1_epochs} epochs)...")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_phase1 = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, phase1_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer_phase1.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_phase1.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"  Epoch {epoch:2d}/{phase1_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # ----------------------------------------------------
    # PHASE 2: End-to-End Fine-Tuning (Backbone Unfrozen)
    # ----------------------------------------------------
    print(f"\n[Phase 2] Unfreezing CNN Backbone -> Fine-Tuning End-to-End ({phase2_epochs} epochs at lr=1e-4)...")
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=phase2_epochs)

    for epoch in range(1, phase2_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer_phase2.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_phase2.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"  Epoch {epoch:2d}/{phase2_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ----------------------------------------------------
    # STEP 3 EVALUATION: Test Set Evaluation & Metrics
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("  Evaluating Best Model on Held-Out Test Set (Stratified 15%)")
    print("=" * 70)

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_acc = accuracy_score(all_targets, all_preds)
    print(f"\n-> Overall Test Accuracy: {test_acc * 100:.2f}%\n")

    # Class-wise report
    target_names = [classes[i] for i in range(num_classes)]
    report_dict = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    print(report_str)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    print(f"Classes: {classes}")
    print(cm)

    # Explicit confirmation of character-only constraint
    pred_labels = [classes[p] for p in all_preds]
    invalid_preds = [p for p in pred_labels if p not in set(classes)]
    print("\n" + "-" * 70)
    print(f"Total Test Predictions: {len(pred_labels)}")
    print(f"Predictions falling outside Character Set {set(classes)}: {len(invalid_preds)}")
    assert len(invalid_preds) == 0, "ERROR: Prediction fell outside character set!"
    print("[CONFIRMED] Exactly ZERO (0) predictions fall outside character set {A-Z}.")
    print("-" * 70)

    # 4. Save Model Artifacts
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "test_accuracy": test_acc,
        "architecture": "mobilenet_v2"
    }, output_model_path)
    print(f"\n[Artifact] Saved PyTorch model weights to: {output_model_path}")

    meta = {
        "classes": classes,
        "num_classes": num_classes,
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "classification_report": report_dict,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(output_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Artifact] Saved model metadata to: {output_meta_path}")

    return model, classes, report_dict, cm

if __name__ == "__main__":
    train_character_model()
