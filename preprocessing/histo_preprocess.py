import cv2
import numpy as np
from skimage import color
from torchvision import transforms

def tissue_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def stain_normalization(img):
    img = img / 255.0
    lab = color.rgb2lab(img)
    lab[..., 0] = (lab[..., 0] - lab[..., 0].mean()) / lab[..., 0].std()
    img_norm = color.lab2rgb(lab)
    return (img_norm * 255).astype(np.uint8)

def extract_patches(img, patch_size=224, stride=224):
    patches = []
    h, w, _ = img.shape
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            if patch.mean() > 20:
                patches.append(patch)
    return patches

histo_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Ensure consistent size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])