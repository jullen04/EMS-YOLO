import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import shutil

# --- KONFIGURATION ---
SAMPLE_SIZE = 50000  # 50 ms totalt tidsfönster (i mikrosekunder)
T = 5                # EMS-YOLO använder 5 tidssteg
H, W = 480, 640      # Din CARLA event-upplösning

# Sökvägar (ANPASSA DESSA TILL DIN DATOR)
# Mappen där du genererade struct och yolo .npy-filer i förra steget
LABELS_BASE_DIR = '/Users/harishaarumuganathan/harisha-programmering/kex/labels_2.0'
# Mappen där dina CARLA-körningar ligger (för att hitta events.h5)
CARLA_BASE_DIR = '/Users/harishaarumuganathan/harisha-programmering/kex/BA_students'
# Var det färdiga träningsdatasetet ska sparas
OUTPUT_DIR = '/Users/harishaarumuganathan/harisha-programmering/kex/dataset_ems'

os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

def render_event_tensor(t_arr, x_arr, y_arr, p_arr, end_time):
    """Skapar en tensor [T, H, W, 3] för ett givet tidsfönster från råa h5-arrays."""
    # EMS-YOLO förväntar sig en grå bakgrund (127) där events är 0 (svart) eller 255 (vit)
    tensor = 127 * np.ones((T, H, W, 3), dtype=np.uint8)
    
    start_time = end_time - SAMPLE_SIZE
    delta_t = SAMPLE_SIZE // T
    
    for i in range(T):
        t0 = start_time + i * delta_t
        t1 = start_time + (i + 1) * delta_t
        
        # Binärsökning för att blixtsnabbt hitta rätt events i de gigantiska arrayerna
        idx0 = np.searchsorted(t_arr, t0)
        idx1 = np.searchsorted(t_arr, t1)
        
        if idx1 > idx0:
            x = x_arr[idx0:idx1].astype(int)
            y = y_arr[idx0:idx1].astype(int)
            p = p_arr[idx0:idx1]
            
            # Säkerhetsfilter: ta bort events som hamnat utanför skärmen
            valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x, y, p = x[valid], y[valid], p[valid]
            
            if len(x) > 0:
                # Polaritet: p > 0 blir vit (255), annars svart (0)
                color = np.where(p > 0, 255, 0)
                
                tensor[i, y, x, 0] = color
                tensor[i, y, x, 1] = color
                tensor[i, y, x, 2] = color
                
    return tensor

def process_dataset():
    # Leta upp alla struct-filer vi skapade tidigare
    struct_files = glob.glob(os.path.join(LABELS_BASE_DIR, 'struct', '**', '*.npy'), recursive=True)
    
    if not struct_files:
        print("Hittade inga struct.npy filer! Har du angett rätt LABELS_BASE_DIR?")
        return

    # Gruppera filerna per CARLA-körning (run_folder)
    runs = {}
    for f in struct_files:
        # Exempel: .../labels_new/struct/Noon_Clear_Town01_xxx/hero_dvs_00/frame_0001_struct.npy
        parts = f.split(os.sep)
        run_name = parts[-3] # Mappnamnet för själva körningen
        if run_name not in runs:
            runs[run_name] = []
        runs[run_name].append(f)

    global_idx = 0

    for run_name, files in runs.items():
        print(f"\nBearbetar körning: {run_name}")
        
        # Hitta motsvarande events.h5 för denna körning
        h5_pattern = os.path.join(CARLA_BASE_DIR, run_name, '**', 'events.h5')
        h5_matches = glob.glob(h5_pattern, recursive=True)
        
        if not h5_matches:
            print(f"  VARNING: Hittade ingen events.h5 för {run_name}. Hoppar över.")
            continue
            
        h5_path = h5_matches[0]
        
        # Ladda in HDF5-filen i minnet för denna körning (mycket snabbare än att läsa från disk i loopen)
        print(f"  Laddar {os.path.basename(h5_path)} till RAM...")
        with h5py.File(h5_path, 'r') as f:
            t_arr = f['events']['t'][:]
            x_arr = f['events']['x'][:]
            y_arr = f['events']['y'][:]
            p_arr = f['events']['p'][:]

        print(f"  Skapar tensors för {len(files)} frames...")
        for struct_path in tqdm(sorted(files)):
            # Läs in vår struct för att hitta tidsstämpeln
            struct_data = np.load(struct_path)
            if len(struct_data) == 0:
                continue # Hoppa över om den mot förmodan är helt tom (och saknar t)
            
            # Alla labels i denna frame har samma tid, så vi tar den första
            end_time = struct_data['t'][0] 
            
            # Generera SNN-tensorn (5x240x304x3)
            event_tensor = render_event_tensor(t_arr, x_arr, y_arr, p_arr, end_time)
            
            # Kontrollera att det faktiskt hände något i tidsfönstret
            if np.all(event_tensor == 127):
                continue # Helt tomt på events (bilen stod still), hoppa över för att undvika brusträning
                
            # Identifiera motsvarande YOLO-label (som vi skapade i förra steget)
            yolo_path = struct_path.replace('struct', 'yolo').replace('_struct.npy', '_yolo.npy')
            if not os.path.exists(yolo_path):
                continue

            # Spara de nya träningsfilerna med ett sekventiellt namn
            out_img_path = os.path.join(OUTPUT_DIR, 'images', f'img_{global_idx:06d}.npy')
            out_lbl_path = os.path.join(OUTPUT_DIR, 'labels', f'img_{global_idx:06d}.npy') # Måste heta samma som bilden för YOLO
            
            np.save(out_img_path, event_tensor)
            shutil.copy(yolo_path, out_lbl_path)
            
            global_idx += 1

    print(f"\nKLAR! Skapade {global_idx} kompletta SNN-träningspar (bild + label) i {OUTPUT_DIR}")

if __name__ == '__main__':
    process_dataset()