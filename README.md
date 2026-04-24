# Prototype-CAM

> **Paper**: *Prototype-CAM: A Prototype-Based Contrastive Class Activation Map for Retinal Layer Segmentation in OCT Images*
[Overview of Prototype-CAM](overview.png)
## Project Structure

```
PrototypeCAM/
├── semanticCAM.py                  # Main 
├── pytorch_grad_cam/
│   ├── prototype.py                # Core 
│   ├── grad_cam.py
│   ├── score_cam.py
│   ├── ...
│   └── utils/
├── model/
│   ├── unet.py                     # U-Net 
│   └── Duke.pth                    # Pre-trained model weights
└── data/
    └── Duke/
        ├── images/
        │   ├── patient_1_B_scan_6.png
        │   └── ...
        ├── label/
        │   ├── patient_1_B_scan_6.png
        │   └── ...
        └── edge_label/
            ├── patient_1_B_scan_6.png
            └── ...
```

### Key Files

| File | Description |
|------|-------------|
| [semanticCAM.py](semanticCAM.py) | Main script for running CAM analysis, supporting multiple CAM methods and two analysis modes |
| [pytorch_grad_cam/prototype.py](pytorch_grad_cam/prototype.py) | Core implementation of the PrototypeCAM algorithm, including prototype initialization (K-Means++), Sinkhorn-Knopp assignment, Mahalanobis-guided contrastive prototype update, and Bayesian posterior scoring |

---

## Analysis Modes

[semanticCAM.py](semanticCAM.py) supports two semantic analysis modes via the `--analysis-mode` parameter, corresponding to two questions defined in the paper:

### `misclassification` — *Why P rather than Q?*

Contrastive explanation for misclassified regions. For pixels whose **ground-truth label is $Q$** but the model **predicted $P$**, this mode highlights what features in the activation map drove the model to choose class $P$ over the correct class $Q$.

- **Positive mask**: pixels where `label == Q` and `prediction == P` (the misclassified region)
- **Negative mask**: pixels where `label == Q` and `prediction == Q` (correctly classified region of the same true class)
- The CAM output reveals the contrastive evidence: why the model favored $P$ rather than $Q$.

```bash
python semanticCAM.py \
    --analysis-mode misclassification \
    --method prototypecam \
    --target-layer Up_conv2 \
    --output-dir cam_mis
```

### `prediction` — *Why P?*

Explanation for predicted regions. For all pixels **predicted as class $P$**, this mode highlights what features support the model's prediction of class $P$ against all other competing classes.

- **Target mask**: pixels where `prediction == P`
- Multi-class contrastive prototypes are constructed for all predicted classes simultaneously, and the CAM score reflects the evidence in favor of $P$ over all competitors.

```bash
python semanticCAM.py \
    --analysis-mode prediction \
    --method prototypecam \
    --target-layer Up_conv2 \
    --output-dir cam_pred
```

---

## Quick Start

### 1. Environment

```bash
pip install torch torchvision opencv-python numpy scipy pillow pandas tqdm
```

### 2. Run PrototypeCAM (Default Configuration)

```bash
python semanticCAM.py \
    --data-root ./data/Duke \
    --model-path ./model/Duke.pth \
    --method prototypecam \
    --analysis-mode prediction \
    --target-layer Up_conv2 \
    --save-type 2 \
    --output-dir cam_results
```

### 3. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--method` | `prototypecam` | CAM method to use |
| `--analysis-mode` | `prediction` | `prediction` (*Why $P$?*) or `misclassification` (*Why $P$ rather than $Q$?*) |
| `--target-layer` | `Up_conv2` | Target layer for visualization |
| `--n-prototypes` | `6` | Number of prototypes per class |
| `--proto-iterations` | `20` | Number of prototype update iterations |
| `--proto-gamma` | `0.9` | Momentum coefficient for prototype update |
| `--proto-temperature` | `0.1` | Temperature for soft assignment |
| `--proto-use-contrastive` | `True` | Enable contrastive learning |
| `--proto-contrastive-weight` | `0.5` | Negative weight $\alpha$ for contrastive scoring |
| `--proto-eta-attract` | `0.2` | Intra-class attraction learning rate $\eta^+$ |
| `--proto-eta-repel` | `2.2` | Inter-class repulsion learning rate $\eta^-$ |
| `--save-type` | `2` | Output type: 1=combined, 2=cam+edge, 3=cam, 4=pure heatmap, 5=npy, 6=heatmap+edge |
| `--restore-size` | `True` | Restore output to original image resolution |
