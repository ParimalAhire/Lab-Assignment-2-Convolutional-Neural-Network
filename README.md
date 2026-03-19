# Lab-Assignment-2-CNN-Transfer-Learning

## Description
This project implements **Convolutional Neural Networks (CNN)** for image classification in two ways:
1. **CNN from Scratch** — custom Sequential block built using Keras Conv2D layers
2. **Transfer Learning** — using pretrained **ResNet50** (trained on ImageNet)

Each approach is tested on two tasks:
- **Binary Classification** — Fire Detection (Fire 🔥 vs No Fire)
- **Multi Classification** — Flowers Recognition (5 flower types 🌸)

All four results are compared at the end to demonstrate the advantage of transfer learning over training from scratch.

---

## Objective
- Build a CNN from scratch using Conv2D, MaxPooling, BatchNormalization and Dense layers
- Apply Transfer Learning using pretrained ResNet50 with frozen base layers
- Run both approaches for binary and multi-class classification
- Compare accuracy, loss curves and confusion matrices across all four models

---

## Algorithm

**CNN from Scratch:**
- 3 Convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
- Flatten → Dense(128, ReLU) → Dropout(0.5)
- Output: Sigmoid (binary) / Softmax (multi-class)
- Optimizer: Adam | Loss: Binary/Categorical Cross-Entropy

**ResNet50 Transfer Learning:**
- Load ResNet50 pretrained on ImageNet — freeze all base layers
- Add custom head: GlobalAveragePooling2D → Dense(128) → Dropout → Output
- Only the custom head is trained
- Optimizer: Adam | Loss: Binary/Categorical Cross-Entropy
- Input preprocessed using `resnet50.preprocess_input`

---

## Datasets

### Binary — Fire Detection
| Property | Details |
|---|---|
| Source | Kaggle — `phylake1337/fire-dataset` |
| Classes | Fire (1) vs No Fire (0) |
| Total Images | ~1,900 |
| Image Size | 128×128 (Scratch) / 224×224 (ResNet50) |
| Split | 80% Train / 20% Test |

### Multi — Flowers Recognition
| Property | Details |
|---|---|
| Source | Kaggle — `alxmamaev/flowers-recognition` |
| Classes | Daisy, Dandelion, Rose, Sunflower, Tulip |
| Total Images | ~4,200 |
| Image Size | 128×128 (Scratch) / 224×224 (ResNet50) |
| Split | 80% Train / 20% Test |

---

## Notebook Structure

```
Step 1  → Import all libraries (once, shared)
Step 2  → Mount Google Drive + set dataset paths
Step 3  → Load & explore Fire dataset (shared)
Step 4  → Load & explore Flowers dataset (shared)
──────────────────────────────────────────────────────
Part 1  → CNN from Scratch — Fire Detection  (Steps 5–7)
Part 2  → CNN from Scratch — Flowers         (Steps 8–10)
Part 3  → ResNet50 Transfer — Fire Detection (Steps 11–14)
Part 4  → ResNet50 Transfer — Flowers        (Steps 15–17)
──────────────────────────────────────────────────────
Part 5  → Compare all 4 models               (Steps 18–20)
──────────────────────────────────────────────────────
Step 21 → Single image — all 4 predict together
```

> **Note:** Imports and datasets are defined **once** and shared across all parts.  
> To change the dataset, only update Steps 3 & 4.

---

## Results

| Model | Task | Accuracy |
|---|---|---|
| CNN from Scratch | Binary (Fire) | 84.50% |
| CNN from Scratch | Multi (Flowers) | 65.05% |
| ResNet50 Transfer Learning | Binary (Fire) | **96.50%** |
| ResNet50 Transfer Learning | Multi (Flowers) | **89.93%** |

### Key Observations
- ResNet50 outperforms CNN Scratch by **+12%** on binary classification
- ResNet50 outperforms CNN Scratch by **+24.88%** on multi-class classification
- The gap is larger on multi-class — pretrained features are more valuable when the problem is harder
- `resnet50.preprocess_input` is essential for correct ResNet50 performance

---

## Model Architecture

### CNN from Scratch
```
Input (128×128×3)
  → Conv2D(32)  + BatchNorm + ReLU + MaxPool(2×2)
  → Conv2D(64)  + BatchNorm + ReLU + MaxPool(2×2)
  → Conv2D(128) + BatchNorm + ReLU + MaxPool(2×2)
  → Flatten
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)    ← Binary
  → Dense(5, Softmax)    ← Multi
```

### ResNet50 Transfer Learning
```
Input (224×224×3) — preprocessed with resnet50.preprocess_input
  → ResNet50 base (frozen — pretrained on ImageNet)
  → GlobalAveragePooling2D
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)    ← Binary
  → Dense(5, Softmax)    ← Multi
```

---

## Google Drive Structure
```
My Drive/
└── DeepLearningLab/
    └── CNN/
        ├── CNN_Assignment2_Drive.ipynb   ← this notebook
        ├── fire_dataset/
        │   ├── fire_images/
        │   └── non_fire_images/
        └── flowers/
            ├── daisy/
            ├── dandelion/
            ├── rose/
            ├── sunflower/
            └── tulip/
```

---

## How to Run

1. Upload this notebook to `My Drive/DeepLearningLab/CNN/`
2. Open in **Google Colab**
3. Enable GPU: Runtime → Change runtime type → **T4 GPU**
4. Run all cells in order
5. At Step 2, click **Allow** when prompted to access Google Drive
6. At Step 21, change `test_on = 'fire'` or `test_on = 'flowers'` to test different images

---

## Requirements

All libraries are pre-installed in Google Colab. No manual installation needed.

```
tensorflow >= 2.x
opencv-python (cv2)
numpy
matplotlib
seaborn
scikit-learn
```

---

## Important Notes

- ✅ **T4 GPU** must be enabled for reasonable training times (~15–25 min total)
- ✅ `preprocess_input` from `resnet50` is required before passing images to ResNet50
- ✅ Images normalized to `[0,1]` for CNN Scratch; multiplied back to `[0,255]` before `preprocess_input` for ResNet50
- ✅ `EarlyStopping` used on all models to prevent overfitting
- ✅ CNN Scratch uses 128×128 input; ResNet50 uses 224×224 input

---

## Applications

**Fire Detection:**
- Forest fire early warning systems
- Smart CCTV fire alerts
- Industrial safety monitoring

**Flower Recognition:**
- Plant identification apps (e.g. PlantNet)
- Botanical research and cataloguing
- E-commerce flower shop automation

**General CNN / Transfer Learning:**
- Medical imaging (X-ray, MRI diagnosis)
- Self-driving car object detection
- Face recognition systems
- Quality control in manufacturing

---

## Conclusion

This assignment implemented CNN-based image classification across four configurations and demonstrated the clear advantage of transfer learning:

1. **CNN Scratch — Binary (Fire):** 84.50% — good result for training from scratch on a small dataset
2. **CNN Scratch — Multi (Flowers):** 65.05% — lower due to 5 similar-looking classes and limited data
3. **ResNet50 — Binary (Fire):** 96.50% — +12% over scratch; pretrained features transfer well
4. **ResNet50 — Multi (Flowers):** 89.93% — +24.88% over scratch; biggest improvement on harder task

**Key takeaway:** Transfer learning with ResNet50 consistently outperforms CNN from scratch, especially on multi-class problems. The pretrained ImageNet features generalize effectively to completely different domains like fire detection and flower recognition, achieving high accuracy with far less training time.

---

**Name:** Parimal Ahire  
**PRN:** 202301040067  
**Course:** Deep Learning Lab
