import cv2
import numpy as np
import os

def test_image(image_path):
    print(f"Tesztelés: {image_path}")
    if not os.path.exists(image_path):
        print(f"  HIBA: A fájl nem létezik!")
        return False
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  HIBA: Nem sikerült betölteni a képet!")
        return False
    
    print(f"  Kép mérete: {img.shape}")
    if len(img.shape) < 3:
        print(f"  HIBA: A kép nem színes (csak {len(img.shape)} dimenziója van)!")
        return False
    
    if img.shape[2] != 4:
        print(f"  HIBA: A kép nem tartalmaz alfa csatornát (csak {img.shape[2]} csatorna)!")
        return False
    
    print(f"  Kép sikeresen betöltve, megfelelő formátumban.")
    return True

# Teszteljük az összes képet
images = ["accessories/hat.png", "accessories/sunglasses.png", 
          "accessories/mustache1.png", "accessories/mustache2.png"]

for img_path in images:
    test_image(img_path)

if not os.path.exists("accessories"):
    os.makedirs("accessories")
    print("'accessories' mappa létrehozva.")

for filename in ["hat.png", "sunglasses.png", "mustache1.png", "mustache2.png"]:
    source_path = filename
    target_path = os.path.join("accessories", filename)
    
    if os.path.exists(source_path) and not os.path.exists(target_path):
        import shutil
        shutil.copy(source_path, target_path)
        print(f"Kép átmásolva: {source_path} -> {target_path}")