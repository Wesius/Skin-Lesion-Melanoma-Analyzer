import os
from tqdm import tqdm
import torch
from torch import device
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report
from torchvision.models import efficientnet_v2_s
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

# CUDA setup
device = torch.device("cuda")
print(f"Using device: {device}")


class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.labels = [1 if img.startswith('mel_') else 0 for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinLesionModel, self).__init__()
        self.efficientnet = efficientnet_v2_s(weights='DEFAULT')
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs=10, patience=3,
                accumulation_steps=4):
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{running_loss / (i + 1):.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_predictions

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                break

    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

    return model


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            chunk_size = 8
            for i in range(0, images.size(0), chunk_size):
                chunk_images = images[i:i + chunk_size]
                chunk_labels = labels[i:i + chunk_size]

                with autocast():
                    outputs = model(chunk_images)
                _, predicted = torch.max(outputs.data, 1)
                total += chunk_labels.size(0)
                correct += (predicted == chunk_labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(chunk_labels.cpu().numpy())

            torch.cuda.empty_cache()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    label_names = ['Non-Melanoma', 'Melanoma']
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    return accuracy


def main():
    # Set paths to the prepared dataset
    train_dir = 'prepared_dataset/train'
    val_dir = 'prepared_dataset/val'
    test_dir = 'prepared_dataset/test'

    # Create datasets
    train_dataset = SkinLesionDataset(train_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(val_dir, transform=val_test_transform)
    test_dataset = SkinLesionDataset(test_dir, transform=val_test_transform)

    # Define training constants
    batch_size = 32
    num_epochs = 5
    patience = 3
    accumulation_steps = 4

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             persistent_workers=True)

    # Print dataset sizes
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # Initialize model and training components
    model = SkinLesionModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    scaler = GradScaler()

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs=num_epochs,
                        patience=patience, accumulation_steps=accumulation_steps)

    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on test set
    accuracy = evaluate_model(model, test_loader)
    print(f"Final Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()