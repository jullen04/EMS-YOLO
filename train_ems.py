# Standalone EMS-YOLO Training Script
# Bypasses the original train_g1.py dataloader issues.
# Directly loads .npy data and trains the spiking YOLO model.

import sys
import os
sys.path.insert(0, 'g1-resnet')
sys.path.insert(0, 'g1-resnet/utils')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import non_max_suppression, xywh2xyxy, box_iou
from utils.metrics import ap_per_class
from tqdm import tqdm
import json
import cv2

#  CONFIG 
DATA_DIR = r'D:\gen1_processed'
EPOCHS = 20
BATCH = 2
IMG_SIZE = 320
DEVICE = 'cuda:0'
LR = 0.001
NC = 2  # number of classes
NAMES = ['car', 'pedestrian']


class Gen1Dataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=320):
        self.img_size = img_size
        split_dir = os.path.join(data_dir, split)
        self.img_files = sorted([
            os.path.join(split_dir, f) for f in os.listdir(split_dir)
            if f.startswith('img_')
        ])
        self.lbl_files = [
            f.replace('img_', 'label_') for f in self.img_files
        ]
        # Filter out pairs where label file doesn't exist
        valid = [(i, l) for i, l in zip(self.img_files, self.lbl_files) if os.path.exists(l)]
        self.img_files = [v[0] for v in valid]
        self.lbl_files = [v[1] for v in valid]
        print(f"  {split}: {len(self.img_files)} samples")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image: (5, 240, 304, 3) -> resize -> (5, 3, img_size, img_size)
        img = np.load(self.img_files[idx])  # (5, 240, 304, 3)
        
        # Resize each timestep
        resized = np.zeros((5, self.img_size, self.img_size, 3), dtype=np.uint8)
        for t in range(5):
            resized[t] = cv2.resize(img[t], (self.img_size, self.img_size))
        
        # (5, H, W, 3) -> (5, 3, H, W) and normalize to float
        resized = resized.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        
        # Load labels: (N, 5) [class, cx, cy, w, h]
        labels = np.load(self.lbl_files[idx])  # (N, 5)
        
        # Add batch index column: (N, 6) [batch_idx, class, cx, cy, w, h]
        if len(labels) > 0:
            out_labels = np.zeros((len(labels), 6), dtype=np.float32)
            out_labels[:, 1:] = labels
        else:
            out_labels = np.zeros((0, 6), dtype=np.float32)
        
        return torch.from_numpy(resized), torch.from_numpy(out_labels)


def collate_fn(batch):
    imgs, labels = zip(*batch)
    # Fix batch index in labels
    for i, l in enumerate(labels):
        l[:, 0] = i
    return torch.stack(imgs, 0), torch.cat(labels, 0)


def compute_map(model, dataloader, device, nc, conf_thres=0.25, iou_thres=0.45):
    # Compute mAP on a dataset
    model.eval()
    stats = []
    
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs = imgs.to(device)
            
            # Forward
            out = model(imgs)
            preds = non_max_suppression(out[0], conf_thres=conf_thres, iou_thres=iou_thres)
            
            # Per image in batch
            for si, pred in enumerate(preds):
                # Get targets for this image
                tgt = targets[targets[:, 0] == si]
                nl = len(tgt)
                tcls = tgt[:, 1].tolist() if nl else []
                
                if len(pred) == 0:
                    if nl:
                        # 2D (n,1) required by ap_per_class which does tp.shape[1]
                        stats.append((torch.zeros(0, 1, dtype=torch.bool),
                                     torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions: move to CPU for metric computation
                # stats are .cpu().numpy() at the end anyway
                predn = pred.clone().cpu()

                # Target boxes: convert from xywh to xyxy (targets already on CPU)
                if nl:
                    tbox = xywh2xyxy(tgt[:, 2:6])  # CPU
                    tbox[:, [0, 2]] *= imgs.shape[-1]
                    tbox[:, [1, 3]] *= imgs.shape[-2]

                    # 2D (n,1): ap_per_class expects tp shape (n_preds, n_iou_thresholds)
                    correct = torch.zeros(predn.shape[0], 1, dtype=torch.bool)
                    if predn.shape[0]:
                        iou = box_iou(predn[:, :4], tbox)  # both CPU
                        for j in range(len(tgt)):
                            # Find best prediction for this target
                            max_iou, max_idx = iou[:, j].max(0)
                            if max_iou > 0.5:
                                if not correct[max_idx]:
                                    correct[max_idx] = True

                    stats.append((correct, predn[:, 4], predn[:, 5], tcls))
                else:
                    stats.append((torch.zeros(predn.shape[0], 1, dtype=torch.bool),
                                 predn[:, 4], predn[:, 5], tcls))
    
    if not stats:
        return {}, 0.0
    
    # Concatenate stats
    stats = [torch.cat(x, 0).cpu().numpy() if isinstance(x[0], torch.Tensor) 
             else np.concatenate(x, 0) for x in zip(*stats)]
    
    results = {}
    if len(stats) and stats[0].any():
        tp, conf, pred_cls, target_cls = stats
        names_dict = {i: n for i, n in enumerate(NAMES)}
        p, r, ap, f1, ap_class = ap_per_class(tp, conf, pred_cls, target_cls, names=names_dict)
        ap50 = ap[:, 0]  # AP@0.5
        mp, mr, map50, mf1 = p.mean(), r.mean(), ap50.mean(), f1.mean()
        
        for i, c in enumerate(ap_class):
            results[NAMES[int(c)]] = {
                'precision': float(p[i]),
                'recall': float(r[i]),
                'ap50': float(ap50[i]),
                'f1': float(f1[i])
            }
        results['mean'] = {
            'precision': float(mp),
            'recall': float(mr),
            'mAP@0.5': float(map50),
            'f1': float(mf1)
        }
        return results, float(map50)
    else:
        return {'mean': {'precision': 0, 'recall': 0, 'mAP@0.5': 0, 'f1': 0}}, 0.0


def print_results_table(results, epoch=None):
    # Print a nice results table
    header = f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'F1':>10}"
    print("\n" + "=" * 60)
    if epoch is not None:
        print(f"Results after epoch {epoch}")
    print(header)
    print("-" * 60)
    for cls_name, metrics in results.items():
        if cls_name == 'mean':
            print("-" * 60)
            cls_name = 'MEAN'
        p = metrics.get('precision', metrics.get('mAP@0.5', 0))
        r = metrics.get('recall', 0)
        m = metrics.get('ap50', metrics.get('mAP@0.5', 0))
        f = metrics.get('f1', 0)
        print(f"{cls_name:<15} {p:>10.4f} {r:>10.4f} {m:>10.4f} {f:>10.4f}")
    print("=" * 60 + "\n")


def main():
    print(f"Loading model...")
    model = Model('g1-resnet/models/resnet34-cat.yaml', ch=3, nc=NC)
    model = model.to(DEVICE)
    
    print(f"Loading datasets from {DATA_DIR}...")
    train_ds = Gen1Dataset(DATA_DIR, 'train', IMG_SIZE)
    val_ds = Gen1Dataset(DATA_DIR, 'val', IMG_SIZE)
    test_ds = Gen1Dataset(DATA_DIR, 'test', IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Attach hyperparameters to model 
    import yaml
    with open('g1-resnet/data/hyps/hyp.scratch.yaml') as f:
        hyp = yaml.safe_load(f)
    model.hyp = hyp
    model.gr = 1.0
    model.nc = NC
    compute_loss = ComputeLoss(model)
    
    # Training loop
    best_map = 0.0
    history = []
    
    print(f"\nStarting training: {EPOCHS} epochs, batch size {BATCH}")
    print(f"Device: {DEVICE}")
    print(f"Model: EMS-YOLO ResNet34, {sum(p.numel() for p in model.parameters())} parameters\n")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, targets in pbar:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        
        # Validate every 5 epochs and on last epoch
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            results, map50 = compute_map(model, val_loader, DEVICE, NC)
            print_results_table(results, epoch + 1)
            
            if map50 > best_map:
                best_map = map50
                torch.save(model.state_dict(), 'best_ems_yolo.pt')
                print(f"  New best mAP@0.5: {map50:.4f} - saved best_ems_yolo.pt")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
        })
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    if os.path.exists('best_ems_yolo.pt'):
        model.load_state_dict(torch.load('best_ems_yolo.pt'))
    
    test_results, test_map = compute_map(model, test_loader, DEVICE, NC)
    print_results_table(test_results)
    
    # Save results to JSON 
    output = {
        'model': 'EMS-YOLO (ResNet34)',
        'dataset': 'Prophesee Gen1 (subset)',
        'classes': NAMES,
        'epochs': EPOCHS,
        'batch_size': BATCH,
        'img_size': IMG_SIZE,
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': len(test_ds),
        'test_results': test_results,
        'training_history': history,
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results.json")
    print(f"Best model saved to best_ems_yolo.pt")
    print(f"\nFor your report - copy this table:")
    print_results_table(test_results)


if __name__ == '__main__':
    main()