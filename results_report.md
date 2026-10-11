# 🫁 Chest X-Ray Pneumonia Classifier — Full Results Report

**Model:** ResNet-18 (Transfer Learning) | **Framework:** PyTorch Lightning  
**Training Strategy:** Feature Extraction (only classifier head unfrozen)  
**Hardware:** Apple MPS (Metal) GPU

---

## 1 · Dataset Overview

| Split | BACTERIAL_PNEUMONIA | NORMAL | VIRAL_PNEUMONIA | Total |
|---|---|---|---|---|
| **Train** | 1,338 | 1,338 | 1,338 | **4,014** |
| **Val** | 148 | 148 | 148 | **444** |

> [!NOTE]
> Dataset is perfectly balanced — 1,338 samples per class in training, 148 per class in validation.

---

## 2 · Sample Training Images

![Random Training Images](/Users/shubhammac/.gemini/antigravity/brain/cdc08953-11a6-4217-8738-da652cd47867/1_random_training_images.png)

---

## 3 · Training Run

| Hyperparameter | Value |
|---|---|
| Max Epochs | 10 |
| Target Val Accuracy (EarlyStopping) | 85% |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-2 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Precision | 16-bit Mixed (AMP) |
| Batch Size | 64 |
| Seed | 15 |

> [!IMPORTANT]
> Training completed in **just 1 epoch** — the EarlyStopping callback triggered immediately because the pre-trained weights were already highly effective, pushing val accuracy above 85% on the very first epoch.

---

## 4 · Final Metrics

| Metric | Value |
|---|---|
| ✅ **Final Validation Accuracy** | **85.14%** |
| 📉 **Final Validation Loss** | **0.4413** |

---

## 5 · Per-Class Accuracy

| Class | Accuracy |
|---|---|
| 🦠 BACTERIAL_PNEUMONIA | **93.92%** |
| 🟢 NORMAL | **82.43%** |
| 🧬 VIRAL_PNEUMONIA | **79.05%** |

> [!TIP]
> The model excels at detecting **Bacterial Pneumonia** (93.92%). Viral Pneumonia is the hardest class — likely because its X-ray patterns can visually overlap with Bacterial Pneumonia.

---

## 6 · Confusion Matrix

![Confusion Matrix](/Users/shubhammac/.gemini/antigravity/brain/cdc08953-11a6-4217-8738-da652cd47867/2_confusion_matrix.png)

---

## 7 · Validation Image Predictions

![Validation Predictions](/Users/shubhammac/.gemini/antigravity/brain/cdc08953-11a6-4217-8738-da652cd47867/3_validation_predictions.png)

---

## 8 · Unseen Image Predictions

![Unseen Predictions](/Users/shubhammac/.gemini/antigravity/brain/cdc08953-11a6-4217-8738-da652cd47867/4_unseen_predictions.png)

| # | True Label | Predicted | Result |
|---|---|---|---|
| 1 | BACTERIAL_PNEUMONIA | BACTERIAL_PNEUMONIA | ✅ |
| 2 | BACTERIAL_PNEUMONIA | BACTERIAL_PNEUMONIA | ✅ |
| 3 | BACTERIAL_PNEUMONIA | BACTERIAL_PNEUMONIA | ✅ |
| 4 | NORMAL | NORMAL | ✅ |
| 5 | NORMAL | NORMAL | ✅ |
| 6 | NORMAL | NORMAL | ✅ |
| 7 | VIRAL_PNEUMONIA | BACTERIAL_PNEUMONIA | ❌ |
| 8 | VIRAL_PNEUMONIA | BACTERIAL_PNEUMONIA | ❌ |
| 9 | VIRAL_PNEUMONIA | NORMAL | ❌ |

**Unseen Accuracy: 6/9 = 66.67%**

> [!WARNING]
> All 3 Viral Pneumonia misclassifications are false negatives — the model predicted them as Bacterial or Normal. This is a known challenge: Viral Pneumonia X-rays often look similar to Bacterial ones, especially in the fine-tuning stage. The two-stage training strategy (fine-tuning `layer3` + `layer4`) in the full model description would improve this.

---

## 9 · Key Takeaways

```
┌─────────────────────────────────────────────────────┐
│  Overall Val Accuracy     : 85.14%   ✅ Target Met  │
│  BACTERIAL_PNEUMONIA Acc  : 93.92%   🏆 Best Class  │
│  NORMAL Acc               : 82.43%                  │
│  VIRAL_PNEUMONIA Acc      : 79.05%   ⚠️ Hardest     │
│  Unseen Images Accuracy   : 66.67%   (6/9 correct)  │
└─────────────────────────────────────────────────────┘
```

All output images saved to `./results/`:
- `1_random_training_images.png`
- `2_confusion_matrix.png`
- `3_validation_predictions.png`
- `4_unseen_predictions.png`
