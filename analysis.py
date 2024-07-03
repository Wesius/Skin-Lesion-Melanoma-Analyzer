import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import SkinLesionModel, SkinLesionDataset
import json
import numpy as np
from tqdm import tqdm

print("Starting analysis. This process may take a while. Please be patient.")

# Create results directory
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SkinLesionModel(num_classes=2).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dir = 'prepared_dataset/test'
test_dataset = SkinLesionDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate model
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating model"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

    return all_labels, all_preds, all_probs

# Get predictions
print("Evaluating model...")
true_labels, predictions, probabilities = evaluate_model(model, test_loader)

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# Classification Report
print("Generating classification report...")
report = classification_report(true_labels, predictions, target_names=['Non-Melanoma', 'Melanoma'])
with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# ROC Curve and AUC
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(true_labels, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
plt.close()

# Precision-Recall Curve
print("Generating Precision-Recall curve...")
precision, recall, _ = precision_recall_curve(true_labels, probabilities)
average_precision = average_precision_score(true_labels, probabilities)

plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
plt.close()

# Learning Curves
print("Generating learning curves...")
with open('training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'learning_curves.png'))
plt.close()

# Generate summary report
print("Generating summary report...")
summary = f"""
Model Performance Summary:

Accuracy: {(cm[0,0] + cm[1,1]) / np.sum(cm):.2f}
Precision (Melanoma): {cm[1,1] / (cm[1,1] + cm[0,1]):.2f}
Recall (Melanoma): {cm[1,1] / (cm[1,1] + cm[1,0]):.2f}
F1-Score (Melanoma): {2 * (cm[1,1] / (cm[1,1] + cm[0,1])) * (cm[1,1] / (cm[1,1] + cm[1,0])) / ((cm[1,1] / (cm[1,1] + cm[0,1])) + (cm[1,1] / (cm[1,1] + cm[1,0]))):.2f}
AUC-ROC: {roc_auc:.2f}
Average Precision: {average_precision:.2f}

Confusion Matrix:
{cm}

See 'classification_report.txt' for detailed metrics.

Graph Descriptions:

1. confusion_matrix.png: 
   Visualizes the model's performance in terms of true positives, true negatives, false positives, and false negatives.

2. roc_curve.png:
   Shows the trade-off between the true positive rate and false positive rate at various classification thresholds.

3. precision_recall_curve.png:
   Illustrates the trade-off between precision and recall for different thresholds.

4. learning_curves.png:
   Displays the model's learning progress in terms of loss and accuracy for both training and validation sets over epochs.
"""

with open(os.path.join(results_dir, 'summary_report.txt'), 'w') as f:
    f.write(summary)

print(f"Analysis complete. Results have been saved in the '{results_dir}' directory.")