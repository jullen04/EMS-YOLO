import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import shutil
import gc  # Importera Garbage Collector för att rensa RAM

# --- KONFIGURATION ---
SAMPLE_SIZE = 50000  
T = 5                
H, W = 480, 640      
TARGET_THRES = "thres_0.15"

LABELS_BASE_DIR = '/Users/harishaarumuganathan/harisha-programmering/kex/labels_2.0'
CARLA_BASE_DIR  = '/Users/harishaarumuganathan/harisha-programmering/kex/BA_students'
OUTPUT_DIR      = '/Users/harishaarumuganathan/harisha-programmering/kex/dataset_ems_new_0.15'

os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

def process_dataset():
    search_pattern = os.path.join(LABELS_BASE_DIR, '**', '*_bbox.npy')
    struct_files = glob.glob(search_pattern, recursive=True)

    if not struct_files:
        print(f"Hittade inga *_bbox.npy filer i {LABELS_BASE_DIR}!")
        return

    runs = {}
    for f in struct_files:
        parts = f.split(os.sep)
        run_name = None
        agent_id = None
        for p in parts:
            if p.startswith("Noon_Clear_"):
                run_name = p
            if p.startswith('hero_segm_'):
                agent_id = p.split('_')[-1]
                
        if run_name is None or agent_id is None: continue
            
        if run_name not in runs: runs[run_name] = {}
        if agent_id not in runs[run_name]: runs[run_name][agent_id] = []
        runs[run_name][agent_id].append(f)

    global_idx = 0

    for run_name, agents in runs.items():
        for agent_id, files in agents.items():
            print(f"\nBearbetar körning: {run_name} | Agent: {agent_id}")

            h5_pattern = os.path.join(CARLA_BASE_DIR, run_name, f'hero_dvs_{agent_id}', '**', f'*{TARGET_THRES}*', 'events.h5')
            h5_matches = glob.glob(h5_pattern, recursive=True)

            if not h5_matches: continue
            h5_path = h5_matches[0]
            
            with h5py.File(h5_path, 'r') as f:
                t_arr = f['events']['t'][:]
                x_arr = f['events']['x'][:]
                y_arr = f['events']['y'][:]
                p_arr = f['events']['p'][:]

            valid_mask = t_arr > 0
            t_arr, x_arr, y_arr, p_arr = t_arr[valid_mask], x_arr[valid_mask], y_arr[valid_mask], p_arr[valid_mask]

            if len(t_arr) == 0: continue

            t_zero = t_arr[0]
            t_arr = t_arr - t_zero

            sort_idx = np.argsort(t_arr, kind='stable')
            t_arr, x_arr, y_arr, p_arr = t_arr[sort_idx], x_arr[sort_idx], y_arr[sort_idx], p_arr[sort_idx]

            no_yolo, empty_tensor, out_of_range = 0, 0, 0
            
            for struct_path in tqdm(sorted(files)):
                struct_data = np.load(struct_path)
                if len(struct_data) == 0: continue

                end_time = int(struct_data['t'][0])
                start_time = max(0, end_time - SAMPLE_SIZE)
                if start_time >= end_time: continue
                if end_time > t_arr[-1]:
                    out_of_range += 1
                    continue

                # --- OPTIMERING 1: Klipp ut 50ms-fönstret EN gång ---
                idx_start = np.searchsorted(t_arr, start_time, side='left')
                idx_end = np.searchsorted(t_arr, end_time, side='right')
                
                t_win = t_arr[idx_start:idx_end]
                x_win = x_arr[idx_start:idx_end]
                y_win = y_arr[idx_start:idx_end]
                p_win = p_arr[idx_start:idx_end]

                delta_t = (end_time - start_time) // T
                tensor = 127 * np.ones((T, H, W, 3), dtype=np.uint8)

                for i in range(T):
                    t0 = start_time + i * delta_t
                    t1 = start_time + (i + 1) * delta_t

                    # --- OPTIMERING 2: Sök bara i det lilla fönstret ---
                    i0 = np.searchsorted(t_win, t0, side='left')
                    i1 = np.searchsorted(t_win, t1, side='right')

                    if i1 <= i0: continue

                    x = x_win[i0:i1]
                    y = y_win[i0:i1]
                    p = p_win[i0:i1]

                    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                    
                    # --- OPTIMERING 3: Konvertera till int BARA för de giltiga (sparar extremt mycket tid) ---
                    x = x[valid].astype(np.int32)
                    y = y[valid].astype(np.int32)
                    p = p[valid]

                    if len(x) == 0: continue

                    color = np.where(p > 0, 255, 0).astype(np.uint8)
                    tensor[i, y, x, 0] = color
                    tensor[i, y, x, 1] = color
                    tensor[i, y, x, 2] = color

                if np.all(tensor == 127):
                    empty_tensor += 1
                    continue

                yolo_path = struct_path.replace('/struct/', '/yolo/').replace('_bbox.npy', '_yolo.npy')
                if not os.path.exists(yolo_path):
                    no_yolo += 1
                    continue

                clean_run_name = run_name.replace(":", "_").replace(" ", "_")
                original_filename = os.path.basename(struct_path)
                frame_str = "".join(filter(str.isdigit, original_filename))
                
                file_id = f"{clean_run_name}_ag{agent_id}_{frame_str}.npy"
                
                out_img_path = os.path.join(OUTPUT_DIR, 'images', file_id)
                out_lbl_path = os.path.join(OUTPUT_DIR, 'labels', file_id)

                np.save(out_img_path, tensor)
                shutil.copy(yolo_path, out_lbl_path)

                global_idx += 1
                
            print(f"  Inget yolo: {no_yolo} | Tom tensor: {empty_tensor} | Utanför räckvidd: {out_of_range}")
            
            # --- OPTIMERING 4: Töm RAM-minnet innan nästa stora h5-fil laddas ---
            del t_arr, x_arr, y_arr, p_arr, t_win, x_win, y_win, p_win
            gc.collect() 

    print(f"\nKLAR! {global_idx} par sparade i {OUTPUT_DIR}")

if __name__ == '__main__':
    process_dataset()