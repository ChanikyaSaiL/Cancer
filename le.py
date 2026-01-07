import os

histo_dir = r"archive (1)"

image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

total_images = 0
folder_wise_count = {}

for root, dirs, files in os.walk(histo_dir):
    count = 0
    for file in files:
        if file.lower().endswith(image_exts):
            count += 1
            total_images += 1
    
    if count > 0:
        folder_wise_count[root] = count

print("\n📊 Image Count Summary\n")

for folder, cnt in folder_wise_count.items():
    print(f"{folder}  →  {cnt} images")

print("\n🧮 TOTAL IMAGES IN DATASET:", total_images)
