import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Sökvägar ---
CARLA_DIR = Path('/Users/harishaarumuganathan/harisha-programmering/kex/yolo_dataset_noon/')
# Den nya mappen som ska skickas till servern för träning
GEN1_FORMAT_DIR = Path('/Users/harishaarumuganathan/harisha-programmering/kex/carla_gen1_format/')

splits = ['train', 'val', 'test']

for split in splits:
    img_dir = CARLA_DIR / 'images' / split
    lbl_dir = CARLA_DIR / 'labels' / split
    out_split_dir = GEN1_FORMAT_DIR / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(img_dir.glob('*.npy'))
    if not npy_files:
        continue
        
    print(f"Konverterar {split} ({len(npy_files)} filer)...")
    
    for i, npy_path in enumerate(tqdm(npy_files)):
        txt_path = lbl_dir / (npy_path.stem + '.txt')
        
        if not txt_path.exists():
            continue
            
        # 1. BILD-KONVERTERING
        # Ladda CARLA-formatet: (T, 2, H, W)
        carla_events = np.load(npy_path)
        T, _, H, W = carla_events.shape
        
        # Skapa Gen1-formatet: (T, H, W, 3) med 127 som bakgrund
        gen1_img = np.full((T, H, W, 3), 127, dtype=np.uint8)
        
        # I CARLA var kanal 0 positiv, kanal 1 negativ
        pos_mask = carla_events[:, 0, :, :] > 0
        neg_mask = carla_events[:, 1, :, :] > 0
        
        # Gen1 färgkodning: 255 = ON, 0 = OFF
        gen1_img[pos_mask] = 255
        gen1_img[neg_mask] = 0
        
        # 2. LABEL-KONVERTERING
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id, cx, cy, w, h = map(float, parts)
                # Gen1 formatet vill ha: [class_id, cx, cy, w, h]
                labels.append([cls_id, cx, cy, w, h])
                
        # Konvertera till float32 array med shape (N, 5)
        labels_arr = np.array(labels, dtype=np.float32)
        
        # 3. SPARA MED RÄTT NAMN
        # Enligt din README ska de heta img_*.npy och label_*.npy inuti split-mappen
        out_img_name = f"img_{i:06d}.npy"
        out_lbl_name = f"label_{i:06d}.npy"
        
        np.save(out_split_dir / out_img_name, gen1_img)
        np.save(out_split_dir / out_lbl_name, labels_arr)

print(f"\nKlar! Din data är nu i exakt det format som train_ems.py förväntar sig.")
print(f"Ladda upp mappen {GEN1_FORMAT_DIR} till din Linux-server.")