import os
import cv2
import torch
from torch.utils.data import Dataset
from preprocessing.histo_preprocess import histo_transform
from preprocessing.photo_preprocess import photo_transform

class MultimodalDataset(Dataset):
    def __init__(self, histo_dir, photo_dir):
        self.samples = []

        for label, cls in enumerate(["Normal", "OSCC"]):
            histo_path = os.path.join(histo_dir, cls)
            photo_path = os.path.join(photo_dir, cls)

            histo_files = sorted([
                os.path.join(histo_path, f)
                for f in os.listdir(histo_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

            photo_files = sorted([
                os.path.join(photo_path, f)
                for f in os.listdir(photo_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

            min_len = min(len(histo_files), len(photo_files))

            for i in range(min_len):
                self.samples.append((histo_files[i], photo_files[i], label))

        if len(self.samples) == 0:
            raise RuntimeError("No valid multimodal samples found.")

        print(f"MultimodalDataset initialized with {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        histo_path, photo_path, label = self.samples[idx]

        histo = cv2.imread(histo_path)
        photo = cv2.imread(photo_path)

        # Robust safety: skip corrupted images
        if histo is None or photo is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        histo = cv2.cvtColor(histo, cv2.COLOR_BGR2RGB)
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

        histo = histo_transform(histo)
        photo = photo_transform(photo)

        return histo, photo, torch.tensor(label, dtype=torch.float32)