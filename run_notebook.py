"""
Full pipeline script derived from C2M4_Assignment.ipynb
Chest X-Ray Pneumonia Classifier using ResNet-18 + PyTorch Lightning
"""

import os
import sys
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix
from torchvision import datasets, transforms, models as tv_models
from PIL import Image
from tqdm.auto import tqdm

import helper_utils

torch.set_float32_matmul_precision('medium')

# Output directory for all saved images
OUT_DIR = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATASET OVERVIEW
# ─────────────────────────────────────────────
data_dir = "./chest_xray/"

print("=" * 50)
print("1. DATASET COUNTS")
print("=" * 50)
helper_utils.display_dataset_count(data_dir)

# ─────────────────────────────────────────────
# 2. DISPLAY RANDOM TRAINING IMAGES
# ─────────────────────────────────────────────
print("\n[Saving] Random training images grid ...")

class_names_ordered = ['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
num_classes_display = len(class_names_ordered)
fig, axs = plt.subplots(num_classes_display, 2, figsize=(7, 7))

for i, class_name in enumerate(class_names_ordered):
    class_path = os.path.join(data_dir + "train", class_name)
    if not os.path.isdir(class_path):
        for j in range(2): axs[i, j].axis('off')
        continue
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 2:
        for j in range(2): axs[i, j].axis('off')
        continue
    selected_images = random.sample(image_files, 2)
    for j, image_file in enumerate(selected_images):
        image_path = os.path.join(class_path, image_file)
        image = Image.open(image_path)
        ax = axs[i, j]
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(class_name, fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_random_training_images.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved → {OUT_DIR}/1_random_training_images.png")

# ─────────────────────────────────────────────
# 3. TRANSFORMS
# ─────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.482, 0.482, 0.482], [0.222, 0.222, 0.222])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.482, 0.482, 0.482], [0.222, 0.222, 0.222])
])

# ─────────────────────────────────────────────
# 4. DATA FUNCTIONS
# ─────────────────────────────────────────────
def create_datasets(train_path, val_path, train_transform, val_transform):
    train_dataset = datasets.ImageFolder(train_path, train_transform)
    val_dataset = datasets.ImageFolder(val_path, val_transform)
    return train_dataset, val_dataset


def load_dataloader(train_dataset, val_dataset, batch_size, is_train_loader):
    if is_train_loader:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return loader


class ChestXRayDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = TRAIN_TRANSFORM
        self.val_transform = VAL_TRANSFORM
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "train")
        val_path = os.path.join(self.data_dir, "val")
        self.train_dataset, self.val_dataset = create_datasets(
            train_path, val_path, self.train_transform, self.val_transform
        )

    def train_dataloader(self):
        return load_dataloader(self.train_dataset, self.val_dataset, self.batch_size, True)

    def val_dataloader(self):
        return load_dataloader(self.train_dataset, self.val_dataset, self.batch_size, False)


# ─────────────────────────────────────────────
# 5. VERIFY DATAMODULE
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. DATAMODULE VERIFICATION")
print("=" * 50)
dm_verify = ChestXRayDataModule(data_dir=data_dir, batch_size=8)
dm_verify.setup()
train_loader_verify = dm_verify.train_dataloader()
val_loader_verify = dm_verify.val_dataloader()

print("--- Training Set ---")
print(f"  Total samples:            {len(dm_verify.train_dataset)}")
print(f"  DataLoader batch size:    {train_loader_verify.batch_size}")
print(f"  DataLoader num batches:   {len(train_loader_verify)}")
print("\n--- Validation Set ---")
print(f"  Total samples:            {len(dm_verify.val_dataset)}")
print(f"  DataLoader batch size:    {val_loader_verify.batch_size}")
print(f"  DataLoader num batches:   {len(val_loader_verify)}")

# ─────────────────────────────────────────────
# 6. MODEL COMPONENTS
# ─────────────────────────────────────────────
def load_resnet18(num_classes, weights_path):
    model = tv_models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def define_optimizer_and_scheduler(model, learning_rate, weight_decay):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    return optimizer, scheduler


class ChestXRayClassifier(pl.LightningModule):
    def __init__(self, model_weights_path, num_classes=3, learning_rate=1e-3, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = load_resnet18(num_classes, model_weights_path)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def configure_optimizers(self):
        optimizer, scheduler = define_optimizer_and_scheduler(
            self.model, self.hparams.learning_rate, self.hparams.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


# ─────────────────────────────────────────────
# 7. EARLY STOPPING
# ─────────────────────────────────────────────
def early_stopping(num_epochs, stop_threshold):
    stop = EarlyStopping(
        monitor="val_acc",
        stopping_threshold=stop_threshold,
        patience=int(num_epochs / 2),
        mode="max"
    )
    return stop


# ─────────────────────────────────────────────
# 8. TRAINING FUNCTION
# ─────────────────────────────────────────────
def run_training(model, data_module, num_epochs, callback, progress_bar=True, dry_run=False):
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=[callback],
        logger=False,
        enable_progress_bar=progress_bar,
        enable_model_summary=False,
        enable_checkpointing=False,
        fast_dev_run=dry_run
    )
    trainer.fit(model, data_module)
    return trainer, model


# ─────────────────────────────────────────────
# 9. FULL TRAINING RUN
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. TRAINING")
print("=" * 50)

pretrained_weights = "./resnet18_chest_xray_classifier_weights.pth"
training_epochs = 10
target_accuracy = 0.85

pl.seed_everything(15)
dm = ChestXRayDataModule(data_dir)
dm.setup()

model = ChestXRayClassifier(model_weights_path=pretrained_weights)
early_stopping_callback = early_stopping(training_epochs, target_accuracy)

trained_trainer, trained_model = run_training(
    model, dm, training_epochs, early_stopping_callback
)
print("\n--- Training Complete ---")

# ─────────────────────────────────────────────
# 10. FINAL METRICS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("4. FINAL METRICS")
print("=" * 50)
final_metrics = trained_trainer.callback_metrics
final_val_acc = final_metrics["val_acc"].item()
final_val_loss = final_metrics["val_loss"].item()
print(f"  Final Validation Accuracy : {final_val_acc:.4f}  ({final_val_acc*100:.2f}%)")
print(f"  Final Validation Loss     : {final_val_loss:.4f}")

# ─────────────────────────────────────────────
# 11. PER-CLASS ACCURACY + CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("5. PER-CLASS ACCURACY & CONFUSION MATRIX")
print("=" * 50)

trained_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = trained_model.to(device)

all_preds = []
all_labels = []

val_loader_with_progress = tqdm(dm.val_dataloader(), desc="Evaluating Model", leave=False)
with torch.no_grad():
    for batch in val_loader_with_progress:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = trained_model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

confmat = MulticlassConfusionMatrix(num_classes=3).to(device)
cm = confmat(all_preds, all_labels)
per_class_acc = cm.diag() / cm.sum(axis=1)
class_names = dm.val_dataset.classes

print("--- Per-Class Accuracy Report ---")
for i, acc in enumerate(per_class_acc):
    print(f"  - Accuracy for class '{class_names[i]}': {acc.item():.4f}  ({acc.item()*100:.2f}%)")

# Plot and save confusion matrix
cm_np = cm.cpu().numpy()
plt.figure(figsize=(8, 6))
sns.heatmap(cm_np, annot=True, fmt='g', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_confusion_matrix.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Confusion matrix saved → {OUT_DIR}/2_confusion_matrix.png")

# ─────────────────────────────────────────────
# 12. RANDOM VALIDATION PREDICTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("6. VALIDATION IMAGE PREDICTIONS")
print("=" * 50)

val_dataset = dm.val_dataset
num_classes = len(class_names)
class_indices = {i: np.where(np.array(val_dataset.targets) == i)[0] for i in range(num_classes)}

random.seed(42)
selected_indices = []
for i in range(num_classes):
    if len(class_indices[i]) >= 2:
        selected_indices.extend(random.sample(list(class_indices[i]), 2))

images_tensor = torch.stack([val_dataset[i][0] for i in selected_indices]).to(device)
labels_tensor = torch.tensor([val_dataset[i][1] for i in selected_indices])

with torch.no_grad():
    outputs = trained_model(images_tensor)
    preds_tensor = torch.argmax(outputs, dim=1)

fig, axs = plt.subplots(num_classes, 2, figsize=(9, 9))
fig.suptitle('Validation Image Predictions', fontsize=15, x=0.45)
mean = np.array([0.482, 0.482, 0.482])
std  = np.array([0.222, 0.222, 0.222])

for i, idx in enumerate(selected_indices):
    ax = axs[i // 2, i % 2]
    img = val_dataset[idx][0].permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.axis('off')
    true_label = class_names[labels_tensor[i]]
    pred_label = class_names[preds_tensor[i]]
    pred_color = 'green' if true_label == pred_label else 'red'
    ax.text(0.5, 1.05, f"True: {true_label}", ha='center', transform=ax.transAxes, fontsize=10)
    ax.text(0.5, -0.08, f"Pred: {pred_label}", ha='center', transform=ax.transAxes,
            fontsize=10, color=pred_color)

plt.tight_layout(rect=[0, 0.03, 0.85, 0.95], h_pad=1.5)
plt.savefig(f"{OUT_DIR}/3_validation_predictions.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved → {OUT_DIR}/3_validation_predictions.png")

# ─────────────────────────────────────────────
# 13. UNSEEN IMAGE PREDICTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("7. UNSEEN IMAGE PREDICTIONS")
print("=" * 50)

unseen_images = [
    ("./unseen/BACTERIAL_PNEUMONIA/person296_bacteria_1394.jpeg",  "BACTERIAL_PNEUMONIA"),
    ("./unseen/BACTERIAL_PNEUMONIA/person441_bacteria_1916.jpeg",  "BACTERIAL_PNEUMONIA"),
    ("./unseen/BACTERIAL_PNEUMONIA/person564_bacteria_2342.jpeg",  "BACTERIAL_PNEUMONIA"),
    ("./unseen/NORMAL/IM-0353-0001.jpeg",                          "NORMAL"),
    ("./unseen/NORMAL/NORMAL2-IM-0866-0001.jpeg",                  "NORMAL"),
    ("./unseen/NORMAL/IM-0633-0001.jpeg",                          "NORMAL"),
    ("./unseen/VIRAL_PNEUMONIA/person1369_virus_2356.jpeg",        "VIRAL_PNEUMONIA"),
    ("./unseen/VIRAL_PNEUMONIA/person1465_virus_2537.jpeg",        "VIRAL_PNEUMONIA"),
    ("./unseen/VIRAL_PNEUMONIA/person1537_virus_2674.jpeg",        "VIRAL_PNEUMONIA"),
]

transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.482, 0.482, 0.482], [0.222, 0.222, 0.222])
])

trained_model.eval()

n_cols = 3
n_rows = 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 12))
fig.suptitle('Unseen Image Predictions', fontsize=15)
correct = 0

for idx, (img_path, true_class) in enumerate(unseen_images):
    row, col = divmod(idx, n_cols)
    ax = axs[row, col]
    try:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform_infer(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = trained_model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
        predicted_class = class_names[pred_idx]
        pred_color = 'green' if predicted_class == true_class else 'red'
        if predicted_class == true_class:
            correct += 1
        print(f"  [{idx+1}] True: {true_class:25s} | Pred: {predicted_class} {'✓' if predicted_class == true_class else '✗'}")
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"True: {true_class.replace('_',' ')}\nPred: {predicted_class.replace('_',' ')}",
                     fontsize=8, color=pred_color)
    except Exception as e:
        ax.axis('off')
        print(f"  [{idx+1}] Error: {e}")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_unseen_predictions.png", dpi=120, bbox_inches='tight')
plt.close()

unseen_acc = correct / len(unseen_images)
print(f"\n  Unseen accuracy: {correct}/{len(unseen_images)} = {unseen_acc:.2%}")
print(f"  Saved → {OUT_DIR}/4_unseen_predictions.png")

# ─────────────────────────────────────────────
# 14. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("8. FINAL SUMMARY")
print("=" * 50)
print(f"  Final Validation Accuracy : {final_val_acc*100:.2f}%")
print(f"  Final Validation Loss     : {final_val_loss:.4f}")
print(f"  Per-Class Accuracy:")
for i, acc in enumerate(per_class_acc):
    print(f"    - {class_names[i]:25s}: {acc.item()*100:.2f}%")
print(f"  Unseen Images Accuracy    : {unseen_acc:.2%}")
print(f"\n  All result images saved to: {OUT_DIR}/")
print("=" * 50)
print("DONE ✓")
