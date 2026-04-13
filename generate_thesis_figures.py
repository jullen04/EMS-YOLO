# Figures — v3 (fixed)

# LATER for CARLA: TO ADD THE DAY+NIGHT MODEL: set MIXED_TRAINED = True and fill the
# MIXED_* variables in the CONFIG block below

import sys, os, warnings
sys.path.insert(0, 'g1-resnet')
sys.path.insert(0, 'g1-resnet/utils')
warnings.filterwarnings('ignore')

import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# scienceplots IEEE 
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass

# something something
matplotlib.rcParams.update({
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     '--',
    'grid.color':         '#888888',
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.06,
    'lines.linewidth':    1.8,
})

# Color
BLUE   = '#1f77b4'
ORANGE = '#d62728'
GREEN  = '#2ca02c'
GRAY   = '#7f7f7f'

# Config 
DATA_DIR = r'D:\gen1_processed'
CKPT     = 'best_ems_yolo.pt'
OUT_DIR  = 'thesis_figures_v2'
IMG_SIZE = 320
DEVICE   = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NAMES    = ['car', 'pedestrian']
os.makedirs(OUT_DIR, exist_ok=True)

# Mixed model, fill when day+night model is trained
MIXED_TRAINED    = False
MIXED_TRAIN_LOSS = []        # list of 20 floats
MIXED_VAL_MAP    = [None]*4  # mAP@0.5 at epochs [5, 10, 15, 20]
MIXED_TEST_RESULT = {}

# Load saved results
with open('results.json') as f:
    saved = json.load(f)

history = saved['training_history']
epochs_all = [h['epoch'] for h in history]
train_loss_all = [h['train_loss'] for h in history]
day_test = saved['test_results']

VAL_EPOCHS  = [5, 10, 15, 20]
DAY_VAL_MAP = [0.0, 0.0, 0.1190, 0.1003]

has_mixed = MIXED_TRAINED and bool(MIXED_TEST_RESULT)

# Helpers
def save(name):
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'))
    print(f'  saved: {name}.pdf / .png')
    plt.close()

# Model
_model = None
def get_model():
    global _model
    if _model is None:
        from models.yolo import Model
        m = Model('g1-resnet/models/resnet34-cat.yaml', ch=3, nc=2)
        m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        _model = m.to(DEVICE).eval()
    return _model

def load_sample(path):
    img = np.load(path)                            # (5,240,304,3)
    raw = img[2].copy()                            # middle timestep for viz
    resized = np.zeros((5, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for t in range(5):
        resized[t] = cv2.resize(img[t], (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(
        resized.transpose(0,3,1,2).astype(np.float32)/255.0
    ).unsqueeze(0).to(DEVICE)
    return tensor, raw

def render_events(frame_hw3):
    # Ternary event frame (0=none,127=OFF,255=ON) -> RGB polarity image
    ch  = frame_hw3[:, :, 0]
    vis = np.zeros((*ch.shape, 3), dtype=np.uint8)
    vis[ch == 255] = [30,  120, 180]   # blue - ON  (positive polarity)
    vis[ch == 127] = [200,  40,  40]   # red - OFF (negative polarity)
    return vis

def xywhn_to_xyxy(box, W, H):
    cx, cy, bw, bh = box
    x1 = int((cx - bw/2)*W);  y1 = int((cy - bh/2)*H)
    x2 = int((cx + bw/2)*W);  y2 = int((cy + bh/2)*H)
    return max(x1,0), max(y1,0), min(x2,W), min(y2,H)

# Full evaluation for PR curve data 
def evaluate_full(conf_thres=0.001, iou_thres=0.45):
    from utils.general import non_max_suppression, xywh2xyxy, box_iou
    from torch.utils.data import Dataset, DataLoader

    class TestDS(Dataset):
        def __init__(self):
            d = os.path.join(DATA_DIR, 'test')
            imgs = sorted([os.path.join(d,f) for f in os.listdir(d) if f.startswith('img_')])
            lbls = [f.replace('img_','label_') for f in imgs]
            valid = [(i,l) for i,l in zip(imgs,lbls) if os.path.exists(l)]
            self.imgs=[v[0] for v in valid]; self.lbls=[v[1] for v in valid]
        def __len__(self): return len(self.imgs)
        def __getitem__(self, idx):
            img = np.load(self.imgs[idx])
            r   = np.zeros((5,IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)
            for t in range(5): r[t]=cv2.resize(img[t],(IMG_SIZE,IMG_SIZE))
            r = r.transpose(0,3,1,2).astype(np.float32)/255.0
            lbl = np.load(self.lbls[idx])
            out = np.zeros((len(lbl),6),dtype=np.float32); out[:,1:]=lbl
            return torch.from_numpy(r), torch.from_numpy(out)

    def collate(batch):
        imgs,lbls=zip(*batch)
        for i,l in enumerate(lbls): l[:,0]=i
        return torch.stack(imgs), torch.cat(lbls)

    loader = DataLoader(TestDS(), batch_size=2, collate_fn=collate, num_workers=0)
    model  = get_model()
    stats  = []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc='  Evaluating (PR curve)', leave=False):
            imgs = imgs.to(DEVICE)
            preds = non_max_suppression(model(imgs)[0],
                                        conf_thres=conf_thres, iou_thres=iou_thres)
            for si, pred in enumerate(preds):
                tgt  = targets[targets[:,0]==si]
                nl   = len(tgt)
                tcls = tgt[:,1].tolist() if nl else []
                if len(pred)==0:
                    if nl: stats.append((torch.zeros(0,1,dtype=torch.bool),
                                         torch.Tensor(),torch.Tensor(),tcls))
                    continue
                predn = pred.clone().cpu()
                if nl:
                    tbox = xywh2xyxy(tgt[:,2:6])
                    tbox[:,[0,2]] *= imgs.shape[-1]
                    tbox[:,[1,3]] *= imgs.shape[-2]
                    correct = torch.zeros(predn.shape[0],1,dtype=torch.bool)
                    if predn.shape[0]:
                        iou = box_iou(predn[:,:4], tbox)
                        for j in range(len(tgt)):
                            mi, mx = iou[:,j].max(0)
                            if mi>0.5 and not correct[mx]: correct[mx]=True
                    stats.append((correct,predn[:,4],predn[:,5],tcls))
                else:
                    stats.append((torch.zeros(predn.shape[0],1,dtype=torch.bool),
                                  predn[:,4],predn[:,5],tcls))
    return [torch.cat(x,0).cpu().numpy() if isinstance(x[0],torch.Tensor)
            else np.concatenate(x,0) for x in zip(*stats)]


def compute_pr_curve(tp, conf, pred_cls, target_cls, cls_id):
    # Smoothed PR curve + AP for one class using 101-point interpolation
    mask   = pred_cls == cls_id
    n_gt   = int((target_cls == cls_id).sum())
    if n_gt == 0 or mask.sum() == 0:
        return None, None, 0.0

    tp_c   = tp[mask, 0].astype(float)
    conf_c = conf[mask]
    order  = np.argsort(-conf_c)
    tp_c   = tp_c[order]
    tpc    = np.cumsum(tp_c)
    fpc    = np.cumsum(1 - tp_c)
    recall    = tpc / (n_gt + 1e-16)
    precision = tpc / (tpc + fpc + 1e-16)

    # 101-point AP (COCO style)
    rec_pts  = np.linspace(0, 1, 101)
    prec_pts = np.interp(rec_pts, recall, precision, left=1.0)
    ap       = float(np.mean(prec_pts))

    # Apply monotone precision envelope (max from right to left).
    # Because its standard display convention in COCO/YOLO papers, it shows the best achievable precision at each recall level,
    for i in range(len(prec_pts) - 2, -1, -1):
        prec_pts[i] = max(prec_pts[i], prec_pts[i + 1])

    return rec_pts, prec_pts, ap

print("Running test-set evaluation for PR curve data...")
raw_stats = evaluate_full(conf_thres=0.001)
if len(raw_stats) == 4:
    tp_all, conf_all, pred_cls_all, target_cls_all = raw_stats
else:
    tp_all = conf_all = pred_cls_all = target_cls_all = np.array([])

#  FIGURE 1: Training Loss
print("\n[Fig 1] Training loss")
fig, ax = plt.subplots(figsize=(5.5, 3.4))

ax.plot(epochs_all, train_loss_all, color=BLUE, lw=2,
        marker='o', ms=3.5, label='Day-only model', zorder=4)

if MIXED_TRAINED and len(MIXED_TRAIN_LOSS) == 20:
    ax.plot(epochs_all, MIXED_TRAIN_LOSS, color=ORANGE, lw=2,
            marker='s', ms=3.5, ls='--', label='Day+Night model', zorder=4)

# Shade eval checkpoints
for ep in VAL_EPOCHS:
    ax.axvline(ep, color='#bbbbbb', lw=0.9, ls=':', zorder=2)
# Single label for all checkpoint lines
ax.axvline(VAL_EPOCHS[0], color='#bbbbbb', lw=0.9, ls=':',
           label='Evaluation checkpoint', zorder=2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('(A) Training Loss over 20 Epochs')
ax.set_xlim(0.5, 20.5)
ax.set_xticks(range(2, 21, 2))
ax.set_ylim(bottom=0)
ax.legend(loc='upper right')
save('fig1_training_loss')

#  FIGURE 2: Validation mAP Progression
print("[Fig 2] Validation mAP progression")
fig, ax = plt.subplots(figsize=(5.0, 3.4))

ax.plot(VAL_EPOCHS, DAY_VAL_MAP, color=BLUE, lw=2,
        marker='D', ms=8, label='Day-only model', zorder=4, clip_on=False)

if MIXED_TRAINED and all(v is not None for v in MIXED_VAL_MAP):
    ax.plot(VAL_EPOCHS, MIXED_VAL_MAP, color=ORANGE, lw=2,
            marker='s', ms=8, ls='--', label='Day+Night model', zorder=4)

# Annotate every checkpoint with its numeric value
for ep, val in zip(VAL_EPOCHS, DAY_VAL_MAP):
    txt   = f'{val:.3f}' if val > 0.0001 else '0.000'
    # push ep=5 right, ep=20 left so they don't run off the axes
    xoff = 6 if ep < 20 else -6
    ha   = 'left' if ep < 20 else 'right'
    ax.annotate(txt, xy=(ep, val),
                xytext=(xoff, 9), textcoords='offset points',
                fontsize=8.5, color=BLUE, fontweight='bold', ha=ha)

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation mAP@0.5')
ax.set_title('(B) Validation mAP@0.5 at Evaluation Checkpoints')
ax.set_xticks(VAL_EPOCHS)
ax.set_ylim(-0.01, 0.22)

if not MIXED_TRAINED:
    ax.annotate('Day-only model', xy=(VAL_EPOCHS[-1], DAY_VAL_MAP[-1]),
                xytext=(0, -22), textcoords='offset points',
                fontsize=9, color=BLUE, ha='center',
                arrowprops=dict(arrowstyle='-', color=BLUE, lw=0.8))
else:
    ax.legend(loc='upper left', fontsize=8.5)
save('fig2_val_map_progression')


#  FIGURE 3: Precision–Recall Curves
print("[Fig 3] Precision-Recall curves")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
fig.subplots_adjust(wspace=0.06)

cls_colors = [BLUE, GREEN]
cls_labels = ['Car', 'Pedestrian']

for ci, (color, label) in enumerate(zip(cls_colors, cls_labels)):
    ax = axes[ci]

    rec, prec, ap = compute_pr_curve(
        tp_all, conf_all, pred_cls_all, target_cls_all, cls_id=ci)

    if rec is not None:
        ax.fill_between(rec, prec, alpha=0.15, color=color, zorder=2)
        ax.plot(rec, prec, color=color, lw=2.2, zorder=4,
                label=f'Day-only  (AP@0.5 = {ap:.3f})')
        # AP annotation: bottom left, away from legend (upper right)
        ax.text(0.04, 0.08, f'AP@0.5 = {ap:.3f}',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=9.5, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor=color,
                          boxstyle='round,pad=0.3', alpha=0.88))
    else:
        ax.text(0.5, 0.5, 'No detections\nin test set',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#888888', style='italic')

    # Iso-F1 contour lines
    r_range = np.linspace(0.01, 1.0, 400)
    for f1_val, lbl_pos in zip([0.2, 0.4, 0.6, 0.8], [0.77, 0.66, 0.55, 0.40]):
        p_iso = f1_val * r_range / (2*r_range - f1_val + 1e-9)
        mask  = (p_iso >= 0) & (p_iso <= 1)
        ax.plot(r_range[mask], p_iso[mask], ':', color='#cccccc', lw=0.9, zorder=1)
        idx = np.argmin(np.abs(r_range[mask] - lbl_pos))
        ax.text(r_range[mask][idx], p_iso[mask][idx] + 0.02,
                f'F\u2081={f1_val}', fontsize=7.5, color='#aaaaaa',
                ha='center')

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('Recall')
    if ci == 0: ax.set_ylabel('Precision')
    ax.set_title(f'({chr(65+ci)}) {label}')
    if rec is not None:
        ax.legend(loc='upper right')

fig.suptitle('Precision\u2013Recall Curves \u2014 Nighttime Test Set',
             fontsize=12, y=1.01)
save('fig3_pr_curves')


#  FIGURE 4: Per-Class Metrics Bar Chart
print("[Fig 4] Per-class metrics")
METRIC_KEYS  = ['precision', 'recall', 'ap50', 'f1']
METRIC_NAMES = ['Precision', 'Recall', 'mAP@0.5', 'F1-Score']

def extract(res, cls):
    m = res.get(cls, {})
    return [m.get(k, 0.0) for k in METRIC_KEYS]

day_car  = extract(day_test, 'car')
day_ped  = extract(day_test, 'pedestrian')
mix_car  = extract(MIXED_TEST_RESULT, 'car')         if has_mixed else None
mix_ped  = extract(MIXED_TEST_RESULT, 'pedestrian')  if has_mixed else None

x  = np.arange(len(METRIC_NAMES))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
fig.subplots_adjust(wspace=0.05)

for ax, cls_label, day_vals, mix_vals in [
    (axes[0], 'Car',        day_car, mix_car),
    (axes[1], 'Pedestrian', day_ped, mix_ped),
]:
    if has_mixed and mix_vals is not None:
        w  = 0.36
        b1 = ax.bar(x-w/2, day_vals, w, color=BLUE,   label='Day-only',
                    zorder=3, edgecolor='white', lw=0.5)
        b2 = ax.bar(x+w/2, mix_vals, w, color=ORANGE, label='Day+Night',
                    zorder=3, edgecolor='white', lw=0.5)
        for b in list(b1)+list(b2):
            h = b.get_height()
            if h > 0.02:
                ax.text(b.get_x()+b.get_width()/2, h+0.012,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)
        ax.legend(loc='upper right')
    else:
        bars = ax.bar(x, day_vals, 0.48, color=BLUE, zorder=3,
                      edgecolor='white', lw=0.5,
                      label='Day-only model')
        for b in bars:
            h = b.get_height()
            if h > 0.02:
                ax.text(b.get_x()+b.get_width()/2, h+0.013,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    # "No detections" note for empty bars
    if all(v < 0.001 for v in day_vals):
        ax.text(0.5, 0.5, 'No detections\nin test set',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#999999', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, fontsize=9)
    ax.set_ylim(0, 1.10)
    if ax is axes[0]: ax.set_ylabel('Score')
    ax.set_title(f'({chr(65+list(axes).index(ax))}) {cls_label}')

fig.suptitle('Detection Performance by Class \u2014 Nighttime Test Set',
             fontsize=12, y=1.01)
save('fig4_per_class_metrics')


#  FIGURE 5: Qualitative Detection Results
print("[Fig 5] Qualitative detections (bounding boxes)...")
from utils.general import non_max_suppression

test_dir  = os.path.join(DATA_DIR, 'test')
all_pairs = sorted([
    (os.path.join(test_dir, f), os.path.join(test_dir, f.replace('img_','label_')))
    for f in os.listdir(test_dir) if f.startswith('img_')
])
with_gt = [(i,l) for i,l in all_pairs if os.path.exists(l) and len(np.load(l))>0]

N_SHOW = min(4, len(with_gt))
model  = get_model()

# Colours as normalised RGB for matplotlib
GT_COL   = np.array([230, 100,  20])/255.   # amber: ground truth
PRED_COL = np.array([ 30, 120, 180])/255.   # blue: prediction

def draw_box(ax, x1, y1, x2, y2, col, label=None, lw=2.5):
    rect = mpatches.FancyBboxPatch((x1, y1), x2-x1, y2-y1,
        boxstyle='square,pad=0', lw=lw,
        edgecolor=col, facecolor='none', zorder=5)
    ax.add_patch(rect)
    if label:
        ax.text(x1+2, max(y1-4, 2), label, fontsize=8, color='white',
                va='bottom',
                bbox=dict(facecolor=col, edgecolor='none',
                          boxstyle='round,pad=1.5', alpha=0.88),
                zorder=6)

fig = plt.figure(figsize=(13, 3.5*N_SHOW))
gs  = gridspec.GridSpec(N_SHOW, 3, figure=fig, hspace=0.04, wspace=0.04)

for row, (img_path, lbl_path) in enumerate(with_gt[:N_SHOW]):
    tensor, raw = load_sample(img_path)
    raw_vis   = render_events(raw)                             # original res
    model_vis = render_events(cv2.resize(raw, (IMG_SIZE, IMG_SIZE)))
    H = W = IMG_SIZE
    labels = np.load(lbl_path)

    with torch.no_grad():
        pred_list = non_max_suppression(model(tensor)[0],
                                        conf_thres=0.25, iou_thres=0.45)
    dets = pred_list[0].cpu().numpy() if len(pred_list[0]) else np.zeros((0,6))

    sid = os.path.basename(img_path).replace('img_','').replace('.npy','')

    # col 0 - raw event frame
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.imshow(raw_vis)
    ax0.set_axis_off()
    lbl0 = f'Event Frame\n(Sample {sid})' if row == 0 else f'Sample {sid}'
    ax0.set_title(lbl0, fontsize=9, pad=3)

    # col 1 - ground truth
    ax1 = fig.add_subplot(gs[row, 1])
    ax1.imshow(model_vis)
    for box in labels:
        cls_id = int(box[0])
        x1,y1,x2,y2 = xywhn_to_xyxy(box[1:], W, H)
        name = NAMES[cls_id] if cls_id < len(NAMES) else str(cls_id)
        draw_box(ax1, x1, y1, x2, y2, GT_COL, label=name)
    ax1.set_axis_off()
    if row == 0: ax1.set_title('Ground Truth', fontsize=9, pad=3)

    # col 2 - predictions
    ax2 = fig.add_subplot(gs[row, 2])
    ax2.imshow(model_vis)
    for det in dets:
        x1,y1,x2,y2,conf,cls_id = det
        cls_id = int(cls_id)
        name   = NAMES[cls_id] if cls_id < len(NAMES) else str(cls_id)
        draw_box(ax2, x1, y1, x2, y2, PRED_COL,
                 label=f'{name} {conf:.2f}')
    if len(dets) == 0:
        ax2.text(W//2, H//2, 'No detections', ha='center', va='center',
                 fontsize=10, color='#dddddd', style='italic')
    ax2.set_axis_off()
    if row == 0: ax2.set_title('Predictions (Day-only model)', fontsize=9, pad=3)

# Legend
legend_elems = [
    mpatches.Patch(color=GT_COL,   label='Ground truth box'),
    mpatches.Patch(color=PRED_COL, label='Predicted box (with confidence)'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=2,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.015),
           framealpha=0.92, edgecolor='#cccccc')
fig.suptitle('Qualitative Detection Results \u2014 Nighttime Test Set',
             fontsize=12, y=1.002)
save('fig5_qualitative_detections')


#  FIGURE 6: Event Frame Polarity Visualization (for Chapter 2)
print("[Fig 6] Event frame visualization")

# Pick 6 diverse examples spread across the test set
n_avail = len(all_pairs)
idxs    = np.linspace(0, n_avail-1, 6, dtype=int)
paths6  = [all_pairs[i][0] for i in idxs]

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
fig.subplots_adjust(hspace=0.06, wspace=0.04)

for i, (ax, path) in enumerate(zip(axes.flat, paths6)):
    _, raw = load_sample(path)
    vis = render_events(raw)
    ax.imshow(vis, aspect='auto')
    ax.set_axis_off()
    ax.set_title(f'Frame {i+1}', fontsize=9, pad=3)

on_patch   = mpatches.Patch(color=np.array([30, 120,180])/255.,
                             label='ON events (positive polarity)')
off_patch  = mpatches.Patch(color=np.array([200, 40, 40])/255.,
                             label='OFF events (negative polarity)')
none_patch = mpatches.Patch(color='black', label='No event (silent pixel)')
fig.legend(handles=[on_patch, off_patch, none_patch],
           loc='lower center', ncol=3, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.025), framealpha=0.92,
           edgecolor='#cccccc')
fig.suptitle('Event Camera Frames \u2014 Polarity Colour Representation\n'
             '(Each frame accumulates events over a 50 ms window)',
             fontsize=12, y=1.01)
save('fig6_event_visualization')


#  FIGURE 7: Combined Training Dashboard (2x2)
print("[Fig 7] Training dashboard")
fig = plt.figure(figsize=(11.5, 7.5))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

# (A) Training loss
ax_l = fig.add_subplot(gs[0,0])
ax_l.plot(epochs_all, train_loss_all, color=BLUE, lw=1.8,
          marker='o', ms=3, label='Day-only')
if MIXED_TRAINED and len(MIXED_TRAIN_LOSS)==20:
    ax_l.plot(epochs_all, MIXED_TRAIN_LOSS, color=ORANGE, lw=1.8,
              marker='s', ms=3, ls='--', label='Day+Night')
for ep in VAL_EPOCHS:
    ax_l.axvline(ep, color='#cccccc', lw=0.8, ls=':')
ax_l.set_xlabel('Epoch'); ax_l.set_ylabel('Loss')
ax_l.set_title('(A) Training Loss'); ax_l.legend(fontsize=8)
ax_l.set_xlim(0.5, 20.5); ax_l.set_xticks(range(2,21,4))
ax_l.set_ylim(bottom=0)

# (B) Validation mAP
ax_v = fig.add_subplot(gs[0,1])
ax_v.plot(VAL_EPOCHS, DAY_VAL_MAP, color=BLUE, lw=2,
          marker='D', ms=7, label='Day-only')
if MIXED_TRAINED and all(v is not None for v in MIXED_VAL_MAP):
    ax_v.plot(VAL_EPOCHS, MIXED_VAL_MAP, color=ORANGE, lw=2,
              marker='s', ms=7, ls='--', label='Day+Night')
for ep, val in zip(VAL_EPOCHS, DAY_VAL_MAP):
    lbl = f'{val:.3f}'
    off = (-22, 7) if ep == 20 else (5, 7)
    ax_v.annotate(lbl, (ep, val), xytext=off,
                  textcoords='offset points', fontsize=8, color=BLUE)
ax_v.set_xlabel('Epoch'); ax_v.set_ylabel('mAP@0.5')
ax_v.set_title('(B) Validation mAP@0.5'); ax_v.legend(fontsize=8)
ax_v.set_xticks(VAL_EPOCHS); ax_v.set_ylim(-0.01, 0.22)

# (C) PR curve - car class (smoothed)
ax_pr = fig.add_subplot(gs[1,0])
rec, prec, ap = compute_pr_curve(tp_all, conf_all, pred_cls_all, target_cls_all, 0)
if rec is not None:
    ax_pr.fill_between(rec, prec, alpha=0.13, color=BLUE)
    ax_pr.plot(rec, prec, color=BLUE, lw=2, label=f'Car (AP={ap:.3f})')
rec2, prec2, ap2 = compute_pr_curve(tp_all, conf_all, pred_cls_all, target_cls_all, 1)
if rec2 is not None:
    ax_pr.fill_between(rec2, prec2, alpha=0.13, color=GREEN)
    ax_pr.plot(rec2, prec2, color=GREEN, lw=2,
               label=f'Pedestrian (AP={ap2:.3f})')
ax_pr.set_xlim(0,1.02); ax_pr.set_ylim(0,1.05)
ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
ax_pr.set_title('(C) Precision\u2013Recall Curve'); ax_pr.legend(fontsize=8)

# (D) Mean metrics bar
ax_b = fig.add_subplot(gs[1,1])
def mean_vals(res):
    m = res.get('mean', {})
    return [m.get('precision',0), m.get('recall',0),
            m.get('mAP@0.5',0),   m.get('f1',0)]

day_m = mean_vals(day_test)
xb    = np.arange(4)
labs  = ['Precision', 'Recall', 'mAP@0.5', 'F1']

if has_mixed:
    mix_m = mean_vals(MIXED_TEST_RESULT)
    b1 = ax_b.bar(xb-0.2, day_m, 0.38, color=BLUE,   label='Day-only',  zorder=3)
    b2 = ax_b.bar(xb+0.2, mix_m, 0.38, color=ORANGE, label='Day+Night', zorder=3)
    for b in list(b1)+list(b2):
        h=b.get_height()
        if h>0.02: ax_b.text(b.get_x()+b.get_width()/2, h+0.01,
                              f'{h:.3f}', ha='center', fontsize=7)
    ax_b.legend(fontsize=8)
else:
    bars = ax_b.bar(xb, day_m, 0.5, color=BLUE, zorder=3)
    for b in bars:
        h=b.get_height()
        if h>0.02: ax_b.text(b.get_x()+b.get_width()/2, h+0.013,
                              f'{h:.3f}', ha='center', fontsize=8,
                              fontweight='bold')
ax_b.set_xticks(xb); ax_b.set_xticklabels(labs, fontsize=8.5)
ax_b.set_ylim(0, 1.05); ax_b.set_ylabel('Score')
ax_b.set_title('(D) Mean Test Performance')

fig.suptitle('Training and Evaluation Summary \u2014 EMS-YOLO SNN\n'
             'Prophesee Gen1 Event Camera Dataset',
             fontsize=12, y=1.005)
save('fig7_training_dashboard')


# Print
print(f'\n{"="*56}')
print(f'All figures saved to: {OUT_DIR}/')
print(f'{"="*56}')
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.pdf'):
        kb = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f'  {f:<42} {kb:>5} KB')
print('\nUse .pdf for Word/LaTeX (vector), .png for Docs/preview.')
