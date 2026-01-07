import os

def count_images(folder):
    return len([f for f in os.listdir(folder)
                if f.lower().endswith(('.png','.jpg','.jpeg'))])

histo_normal = count_images("archive (1)/test/Normal")
histo_oscc   = count_images("archive (1)/test/OSCC")

photo_normal = count_images("photo_split/test/Normal")
photo_oscc   = count_images("photo_split/test/OSCC")

print("\n--- HISTOPATHOLOGY TEST SET ---")
print("Normal:", histo_normal)
print("OSCC  :", histo_oscc)
print("Total :", histo_normal + histo_oscc)

print("\n--- PHOTO TEST SET ---")
print("Normal:", photo_normal)
print("OSCC  :", photo_oscc)
print("Total :", photo_normal + photo_oscc)
