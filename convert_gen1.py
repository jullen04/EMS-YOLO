import sys, os, shutil
import numpy as np

# Copy the Prophesee loader files locally and fix relative imports
src = r'C:\Users\julia\prophesee-automotive-dataset-toolbox\src\io'
for f in ['psee_loader.py', 'dat_events_tools.py', 'npy_events_tools.py']:
    shutil.copy(os.path.join(src, f), f)

for f in ['psee_loader.py']:
    txt = open(f).read()
    txt = txt.replace('from . import dat_events_tools', 'import dat_events_tools')
    txt = txt.replace('from . import npy_events_tools', 'import npy_events_tools')
    open(f, 'w').write(txt)

from psee_loader import PSEELoader

T = 5
SAMPLE_SIZE = 250000
H, W = 240, 304
INPUT = r'D:\gen1_raw\detection_dataset_duration_60s_ratio_1.0\train'
OUTPUT = r'D:\gen1_processed'

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT, split), exist_ok=True)

# Find all file pairs
files = []
for f in os.listdir(INPUT):
    if f.endswith('_td.dat'):
        dat = os.path.join(INPUT, f)
        npy = dat.replace('_td.dat', '_bbox.npy')
        if os.path.exists(npy):
            files.append((dat, npy))

print(f"Found {len(files)} file pairs")

# Split: 10 train, 3 val, 2 test
np.random.seed(42)
np.random.shuffle(files)
splits = {
    'train': files[:10],
    'val': files[10:13],
    'test': files[13:15],
}

total = 0
for split, pairs in splits.items():
    count = 0
    for fi, (dat_path, npy_path) in enumerate(pairs):
        print(f"[{split}] Processing file {fi+1}/{len(pairs)}: {os.path.basename(dat_path)}")
        video = PSEELoader(dat_path)
        boxes = np.load(npy_path)
        if 'ts' in boxes.dtype.names:
            boxes.dtype.names = [n if n != 'ts' else 't' for n in boxes.dtype.names]

        unique_ts = np.unique(boxes['t'])
        for ti, ts in enumerate(unique_ts):
            if ti >= 30:
                break
            ts_boxes = boxes[boxes['t'] == ts]
            try:
                video.seek_time(max(0, int(ts) - SAMPLE_SIZE))
            except:
                continue

            img = 127 * np.ones((T, H, W, 3), dtype=np.uint8)
            for step in range(T):
                try:
                    ev = video.load_delta_t(SAMPLE_SIZE // T)
                except:
                    continue
                if len(ev) > 0:
                    xs = ev['x'].astype(int)
                    ys = ev['y'].astype(int)
                    ps = ev['p']
                    v = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                    img[step, ys[v], xs[v], :] = (255 * ps[v, None]).astype(np.uint8)

            # Convert boxes to YOLO format
            labels = []
            for b in ts_boxes:
                x, y, w, h = float(b['x']), float(b['y']), float(b['w']), float(b['h'])
                if w <= 0 or h <= 0:
                    continue
                cls = int(b['class_id'])
                cx = (x + w/2) / W
                cy = (y + h/2) / H
                labels.append([cls, cx, cy, w/W, h/H])

            if not labels:
                continue

            labels = np.array(labels, dtype=np.float32)
            np.save(os.path.join(OUTPUT, split, f'img_{split}_{fi}_{ti}.npy'), img)
            np.save(os.path.join(OUTPUT, split, f'label_{split}_{fi}_{ti}.npy'), labels)
            count += 1

    print(f"{split}: {count} samples")
    total += count

print(f"\nDone! {total} total samples in {OUTPUT}")
print(f"\nNext: edit g1-resnet\\data\\gen1.yaml, set path to: {OUTPUT}")
print(f"Then run: python g1-resnet\\train_g1.py --weights \"\" --img 320 --batch 2 --epochs 20 --device 0")