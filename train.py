import torch
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from models.multimodal_model import MultimodalNet
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = MultimodalDataset(
    histo_dir="archive (1)/train",
    photo_dir="photo_split/train"
)

val_dataset = MultimodalDataset(
    histo_dir="archive (1)/val",
    photo_dir="photo_split/val"
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = MultimodalNet().to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

os.makedirs("models/saved_models", exist_ok=True)

for epoch in range(30):
    model.train()
    train_loss = 0

    for histo, photo, labels in train_loader:
        histo, photo = histo.to(DEVICE), photo.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)   # ← force [B,1]

        out = model(histo, photo)                 # ← no squeeze
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for histo, photo, labels in val_loader:
            histo, photo = histo.to(DEVICE), photo.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            out = model(histo, photo)
            val_loss += criterion(out, labels).item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "models/saved_models/multimodal_model.pth")
print("Model saved.")
