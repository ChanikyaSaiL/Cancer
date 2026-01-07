import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from dataset import MultimodalDataset
from models.multimodal_model import MultimodalNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset = MultimodalDataset(
    histo_dir="archive (1)/test",
    photo_dir="photo_split/test"
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = MultimodalNet().to(DEVICE)
model.load_state_dict(torch.load("models/saved_models/multimodal_model.pth", map_location=DEVICE))
model.eval()

all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for histo, photo, labels in test_loader:
        histo, photo = histo.to(DEVICE), photo.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(histo, photo)
        labels = labels.unsqueeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

print("\nAccuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds))
print("Recall:", recall_score(all_labels, all_preds))
print("F1:", f1_score(all_labels, all_preds))
print("AUC:", roc_auc_score(all_labels, all_probs))
print("\nConfusion Matrix\n", confusion_matrix(all_labels, all_preds))
