"""
Train U-Net (segmentation lesion only)

Chạy từ project root:

    python -m src.train.train_unet

Có hỗ trợ:
- Progress bar cho train/val/test bằng tqdm
- Đọc num_epochs từ biến môi trường EPOCHS (mặc định 50)
- Ưu tiên dùng splits_by_case nếu tồn tại, nếu không sẽ dùng splits
- Log ra CSV + lưu checkpoint best/last
- Flush CSV từng epoch và sync log + checkpoint + PNG sang DRIVE_ROOT (nếu có)
"""

import os
import sys

# === Thiết lập PROJECT_ROOT để luôn import được "src.***" ===
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.unet import UNet
from src.data.dataset_ct import LungCTMultiTaskDataset
from src.utils.metrics import dice_coefficient, iou_score

# Nếu muốn sync sang thư mục backup (local hoặc Google Drive folder),
# đặt env, ví dụ trong notebook/cmd:
#   set DRIVE_ROOT=D:\Backup\multi_task_ct   (Windows)
#   export DRIVE_ROOT=/path/to/backup        (Linux/Mac)
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", None)


# ==================== SYNC UTILS ====================

def sync_to_drive(project_root, paths):
    """
    Copy các file trong `paths` sang thư mục DRIVE_ROOT,
    giữ nguyên cấu trúc tương đối so với project_root.

    Ví dụ:
        project_root = D:\Code\PythonProject\Advanced\multi_task_ct
        src = D:\Code\PythonProject\Advanced\multi_task_ct\checkpoints\unet_seg_best.pth
        DRIVE_ROOT = D:\Backup\multi_task_ct

        -> dst = D:\Backup\multi_task_ct\checkpoints\unet_seg_best.pth
    """
    if DRIVE_ROOT is None:
        return

    for src in paths:
        if not src:
            continue
        if not os.path.exists(src):
            continue

        rel = os.path.relpath(src, project_root)
        dst = os.path.join(DRIVE_ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


# ==================== LOSS (Dice + BCE) ====================

class DiceLoss(nn.Module):
    """Soft Dice Loss cho segmentation (dùng trực tiếp trên xác suất 0-1)."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred, target: [B,1,H,W]
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class SegCombinedLoss(nn.Module):
    """
    Kết hợp Dice Loss và Binary Cross Entropy cho segmentation.
    pred đã sigmoid trong UNet, nên dùng BCELoss (không phải WithLogits).
    """
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce, dice, bce


# ==================== TRAIN / EVAL LOOP ====================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = len(loader)

    for batch in tqdm(loader, desc=f"Train {epoch}", ncols=100):
        imgs = batch["image"].to(device)   # [B,1,H,W]
        masks = batch["mask"].to(device)   # [B,1,H,W]

        optimizer.zero_grad()
        pred_seg = model(imgs)             # [B,1,H,W], đã sigmoid
        loss, loss_dice, loss_bce = criterion(pred_seg, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(pred_seg.detach(), masks)
        total_iou  += iou_score(pred_seg.detach(), masks)

    return (
        total_loss / n_batches,
        total_dice / n_batches,
        total_iou / n_batches,
    )


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch, phase="Val"):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = len(loader)

    for batch in tqdm(loader, desc=f"{phase} {epoch}", ncols=100):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        pred_seg = model(imgs)
        loss, _, _ = criterion(pred_seg, masks)

        total_loss += loss.item()
        total_dice += dice_coefficient(pred_seg, masks)
        total_iou  += iou_score(pred_seg, masks)

    return (
        total_loss / n_batches,
        total_dice / n_batches,
        total_iou / n_batches,
    )


# ==================== CSV LOGGER ====================

def init_csv_logger(csv_path):
    """
    Tạo file CSV log nếu chưa tồn tại, ghi header.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_dice", "train_iou",
                "val_loss",   "val_dice",   "val_iou",
                "lr",
            ])
            f.flush()
            os.fsync(f.fileno())


def append_csv_log(csv_path, epoch, train_stats, val_stats, lr):
    """
    Ghi thêm 1 dòng log cho mỗi epoch và flush ngay.
    """
    tr_loss, tr_dice, tr_iou = train_stats
    va_loss, va_dice, va_iou = val_stats
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{tr_loss:.6f}", f"{tr_dice:.6f}", f"{tr_iou:.6f}",
            f"{va_loss:.6f}", f"{va_dice:.6f}", f"{va_iou:.6f}",
            f"{lr:.8f}",
        ])
        f.flush()
        os.fsync(f.fileno())


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


# ==================== MAIN ====================

def main():
    # ====== config ======
    project_root = PROJECT_ROOT
    data_root = os.path.join(project_root, "data", "processed", "covid_normal")
    meta_csv = os.path.join(data_root, "meta.csv")

    # Ưu tiên dùng splits_by_case (chia theo bệnh nhân), nếu không có thì dùng splits cũ
    splits_dir_case = os.path.join(data_root, "splits_by_case")
    splits_dir_simple = os.path.join(data_root, "splits")
    if os.path.isdir(splits_dir_case):
        splits_dir = splits_dir_case
    else:
        splits_dir = splits_dir_simple

    train_txt = os.path.join(splits_dir, "train.txt")
    val_txt   = os.path.join(splits_dir, "val.txt")
    test_txt  = os.path.join(splits_dir, "test.txt")

    batch_size = 8
    num_epochs = int(os.environ.get("EPOCHS", 50))  # export EPOCHS=1 để test nhanh
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Project root:", project_root)
    print("Data root:", data_root)
    print("Meta CSV:", meta_csv)
    print("Splits dir:", splits_dir)
    print("Num epochs:", num_epochs)
    if DRIVE_ROOT is not None:
        print("DRIVE_ROOT sync to:", DRIVE_ROOT)
    else:
        print("DRIVE_ROOT not set -> không sync backup.")

    # folders for checkpoint + logs
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "unet_seg_best.pth")
    last_ckpt_path = os.path.join(ckpt_dir, "unet_seg_last.pth")

    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_csv = os.path.join(logs_dir, "unet_seg_log.csv")
    init_csv_logger(log_csv)

    # ====== datasets & loaders ======
    # Dùng lại LungCTMultiTaskDataset nhưng chỉ cần image + mask, bỏ qua label
    train_set = LungCTMultiTaskDataset(
        data_root=data_root,
        split_txt=train_txt,
        meta_csv=meta_csv,
        use_roi=True,
    )
    val_set = LungCTMultiTaskDataset(
        data_root=data_root,
        split_txt=val_txt,
        meta_csv=meta_csv,
        use_roi=True,
    )
    test_set = LungCTMultiTaskDataset(
        data_root=data_root,
        split_txt=test_txt,
        meta_csv=meta_csv,
        use_roi=True,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    # ====== model, loss, optim ======
    model = UNet(in_channels=1, out_channels=1).to(device)
    print("Model:", model.__class__.__name__)
    print("Model parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = SegCombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_dice = 0.0

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
    }

    # ====== training loop ======
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        va_loss, va_dice, va_iou = eval_one_epoch(
            model, val_loader, criterion, device, epoch, phase="Val"
        )

        scheduler.step(va_loss)
        cur_lr = get_current_lr(optimizer)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)

        print(
            f"Train - loss: {tr_loss:.4f}, dice: {tr_dice:.4f}, iou: {tr_iou:.4f}"
        )
        print(
            f"Val   - loss: {va_loss:.4f}, dice: {va_dice:.4f}, "
            f"iou: {va_iou:.4f}, lr: {cur_lr:.6e}"
        )

        # --- LƯU LOG CSV MỖI EPOCH ---
        append_csv_log(
            log_csv,
            epoch,
            train_stats=(tr_loss, tr_dice, tr_iou),
            val_stats=(va_loss, va_dice, va_iou),
            lr=cur_lr,
        )

        # --- LƯU CHECKPOINT "LAST" MỖI EPOCH ---
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
            },
            last_ckpt_path,
        )

        # --- LƯU CHECKPOINT "BEST" KHI VAL DICE TĂNG ---
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_dice": best_val_dice,
                },
                best_ckpt_path,
            )
            print(f"  -> Saved BEST model (val dice={best_val_dice:.4f})")

        # --- SYNC SANG DRIVE MỖI EPOCH ---
        sync_to_drive(
            project_root,
            [log_csv, last_ckpt_path, best_ckpt_path],
        )

    # ====== plot curves (từ history trong RAM, nếu bị ngắt thì xem CSV) ======
    epochs = range(1, len(history["train_loss"]) + 1)
    png_path = os.path.join(project_root, "training_curves_unet_seg.png")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_dice"], label="train_dice")
    plt.plot(epochs, history["val_dice"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.title("Dice")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # sync png
    sync_to_drive(project_root, [png_path])

    # ====== evaluate best model on test ======
    print("\n===== Evaluate BEST model on TEST =====")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    te_loss, te_dice, te_iou = eval_one_epoch(
        model, test_loader, criterion, device, epoch=0, phase="Test"
    )
    print(
        f"Test - loss: {te_loss:.4f}, dice: {te_dice:.4f}, iou: {te_iou:.4f}"
    )

    # sync lại lần cuối
    sync_to_drive(
        project_root,
        [log_csv, last_ckpt_path, best_ckpt_path, png_path],
    )


if __name__ == "__main__":
    main()
