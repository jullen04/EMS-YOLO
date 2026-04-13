# EMS-YOLO 

> **Fork of [BICLab/EMS-YOLO](https://github.com/BICLab/EMS-YOLO)**
>
> Used for the KTH Bachelor thesis:
> *"A Comparative Study of Day-Only versus Day-and-Night Training for Traffic Sign Detection in Spiking Neural Networks"*
> Julia Lohman & Harisha Arumuganathan — KTH School of Electrical Engineering and Computer Science, 2026

---

## What this fork does

This project trains a **Spiking Neural Network (SNN)** to detect traffic signs in data from an event-based camera (DVS). The model is EMS-YOLO, a directly-trained spiking object detector based on a ResNet-34 backbone.

The thesis investigates whether training on a mix of daytime and nighttime data produces a more robust detector than training on daytime data alone, when tested on a nighttime scene the model has never seen.

The original EMS-YOLO codebase was modified to:
- Add a custom training script that works with our preprocessed data format
- Fix several bugs in the evaluation pipeline
- Add scripts for generating publication-quality thesis figures

---

## What was added to the original repo

| File                         | Description                                                                                                                                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_ems.py`               | Standalone training script. Replaces the original `train_g1.py` which was incompatible with our data format. Includes a custom dataloader, training loop, evaluation function, and results export. |
| `generate_thesis_figures.py` | Generates all thesis figures (training loss, validation mAP, precision-recall curves, per-class metrics, qualitative detections, event frame visualisation). Outputs PDF and PNG.                  |
| `convert_gen1.py`            | Converts raw Prophesee Gen1 event data into the `.npy` format used by the training script.                                                                                                         |
| `utils/datasets_g1T.py`      | Custom dataset utilities for the Gen1 format.                                                                                                                                                      |
| `results.json`               | Saved metrics from the completed 20-epoch validation run on the Gen1 subset.                                                                                                                       |

---

## Environment setup

Tested on **Windows 11**, **NVIDIA RTX 4070**, CUDA 11.3.

```bash
conda create -n ems python=3.8
conda activate ems

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install spikingjelly==0.0.0.0.14
pip install opencv-python numpy tqdm pyyaml matplotlib scienceplots
```

---

## Data

Data is from the [Prophesee Gen1 Automotive Detection Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/). A small subset was extracted from `train_a.7z` for pipeline validation.

Each sample consists of two `.npy` files:
- `img_*.npy` — shape `(5, 240, 304, 3)`, uint8. Five accumulated event frames. Pixel values are ternary: `0` = no event, `127` = OFF event, `255` = ON event.
- `label_*.npy` — shape `(N, 5)`, float32. One row per object: `[class_id, cx, cy, w, h]` normalised to image dimensions.

Expected folder structure:
```
D:\gen1_processed\
    train\
        img_*.npy
        label_*.npy
    val\
        img_*.npy
        label_*.npy
    test\
        img_*.npy
        label_*.npy
```

To change the data path, edit the `DATA_DIR` variable at the top of `train_ems.py`.

---

## Training

```bash
conda activate ems
cd C:\path\to\EMS-YOLO
python train_ems.py
```

Training runs for 20 epochs by default. Key parameters at the top of `train_ems.py`:

| Parameter  | Default | Description               |
| ---------- | ------- | ------------------------- |
| `EPOCHS`   | 20      | Number of training epochs |
| `BATCH`    | 2       | Batch size                |
| `LR`       | 0.001   | Learning rate (Adam)      |
| `IMG_SIZE` | 320     | Input resolution (square) |
| `NC`       | 2       | Number of classes         |

After training, results are saved to `results.json` and the best model checkpoint to `best_ems_yolo.pt`.

---

## Generate thesis figures

```bash
python generate_thesis_figures.py
```

Figures are saved to `thesis_figures_v2/` as both `.pdf` (vector, for documents) and `.png` (300 dpi, for preview).

When the day+night model is trained, set `MIXED_TRAINED = True` and fill in the `MIXED_*` variables at the top of the script. All figures will automatically update to show both models side by side.

---

## Results (Gen1 validation run)

Results on the nighttime test set (30 samples, best checkpoint at epoch 15):

| Class      | Precision | Recall | mAP@0.5 | F1-Score |
| ---------- | --------- | ------ | ------- | -------- |
| Car        | 0.541     | 0.394  | 0.457   | 0.456    |
| Pedestrian | 0.000     | 0.000  | 0.000   | 0.000    |

These results are from the pipeline validation run on real Gen1 data. The main thesis experiments use CARLA-generated event data and will be added here when complete.

---

## Original paper

This repo is based on:

> Su et al., *"Deep Directly-Trained Spiking Neural Networks for Object Detection"*, ICCV 2023.
> [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Su_Deep_Directly-Trained_Spiking_Neural_Networks_for_Object_Detection_ICCV_2023_paper.html) · [Original repo](https://github.com/BICLab/EMS-YOLO)

```bibtex
@inproceedings{su2023deep,
  title={Deep Directly-Trained Spiking Neural Networks for Object Detection},
  author={Su, Qiaoyi and Chou, Yuhong and Hu, Yifan and Li, Jianing and Mei, Shijie and Zhang, Ziyang and Li, Guoqi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6555--6565},
  year={2023}
}
```
