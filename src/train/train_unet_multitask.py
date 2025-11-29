"""
Train multi-task U-Net (seg lesion + cls COVID/NORMAL)

Chạy từ project root:

    python -m src.train.train_unet_multitask

Có hỗ trợ:
- Progress bar cho train/val/test bằng tqdmtrain_loader = DataLoader(train_set, batch_s
- Đọc num_epochs từ biến môi trường EPOCHS (mặc định 50)
- Ưu tiên dùng splits_by_case nếu tồn tại, nếu không sẽ dùng splits
"""

import os
import sys
sys.path.append(".")

import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.unet import UNetMultiTask
from src.data.dataset_ct import LungCTMultiTaskDataset
from src.losses.losses import MultiTaskLoss
from src.utils.metrics import dice_coefficient, iou_score, classification_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    n_batches = len(loader)

    for batch in tqdm(loader, desc=f"Train {epoch}", ncols=100):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        pred_seg, pred_cls = model(imgs)
        loss, loss_seg, loss_cls = criterion(pred_seg, masks, pred_cls, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(pred_seg.detach(), masks)
        total_iou  += iou_score(pred_seg.detach(), masks)
        total_acc  += classification_metrics(pred_cls.detach(), labels)

    return (total_loss / n_batches,
            total_dice / n_batches,
            total_iou / n_batches,
            total_acc / n_batches)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch, phase="Val"):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    n_batches = len(loader)

    for batch in tqdm(loader, desc=f"{phase} {epoch}", ncols=100):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        pred_seg, pred_cls = model(imgs)
        loss, _, _ = criterion(pred_seg, masks, pred_cls, labels)

        total_loss += loss.item()
        total_dice += dice_coefficient(pred_seg, masks)
        total_iou  += iou_score(pred_seg, masks)
        total_acc  += classification_metrics(pred_cls, labels)

    return (total_loss / n_batches,
            total_dice / n_batches,
            total_iou / n_batches,
            total_acc / n_batches)


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
                "train_loss", "train_dice", "train_iou", "train_acc",
                "val_loss", "val_dice", "val_iou", "val_acc",
                "lr",
            ])


def append_csv_log(csv_path, epoch, train_stats, val_stats, lr):
    """
    Ghi thêm 1 dòng log cho mỗi epoch.
    """
    tr_loss, tr_dice, tr_iou, tr_acc = train_stats
    va_loss, va_dice, va_iou, va_acc = val_stats
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{tr_loss:.6f}", f"{tr_dice:.6f}", f"{tr_iou:.6f}", f"{tr_acc:.6f}",
            f"{va_loss:.6f}", f"{va_dice:.6f}", f"{va_iou:.6f}", f"{va_acc:.6f}",
            f"{lr:.8f}",
        ])


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def main():
    # ====== config ======
    project_root = os.path.abspath(".")
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
    print("Splits dir:", splits_dir)
    print("Num epochs:", num_epochs)

    # folders for checkpoint + logs
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "unet_multitask_best.pth")
    last_ckpt_path = os.path.join(ckpt_dir, "unet_multitask_last.pth")

    logs_dir = os.path.join(project_root, "logs")
    log_csv = os.path.join(logs_dir, "unet_multitask_log.csv")
    init_csv_logger(log_csv)

    # ====== datasets & loaders ======
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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)


    # ====== model, loss, optim ======
    model = UNetMultiTask(in_channels=1, num_seg_classes=1, num_cls_classes=2).to(device)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    criterion = MultiTaskLoss(w_seg=1.0, w_cls=1.0, dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_dice = 0.0

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_acc":  [], "val_acc": [],
    }

    # ====== training loop ======
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        tr_loss, tr_dice, tr_iou, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        va_loss, va_dice, va_iou, va_acc = eval_one_epoch(
            model, val_loader, criterion, device, epoch, phase="Val"
        )

        scheduler.step(va_loss)
        cur_lr = get_current_lr(optimizer)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Train   - loss: {tr_loss:.4f}, dice: {tr_dice:.4f}, "
              f"iou: {tr_iou:.4f}, acc: {tr_acc:.4f}")
        print(f"Val     - loss: {va_loss:.4f}, dice: {va_dice:.4f}, "
              f"iou: {va_iou:.4f}, acc: {va_acc:.4f}, lr: {cur_lr:.6e}")

        # --- LƯU LOG CSV MỖI EPOCH ---
        append_csv_log(
            log_csv,
            epoch,
            train_stats=(tr_loss, tr_dice, tr_iou, tr_acc),
            val_stats=(va_loss, va_dice, va_iou, va_acc),
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

    # ====== plot curves (từ history trong RAM, nếu bị ngắt thì xem CSV) ======
    epochs = range(1, len(history["train_loss"]) + 1)

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
    plt.savefig(os.path.join(project_root, "training_curves_multitask.png"), dpi=150)
    plt.close()

    # ====== evaluate best model on test ======
    print("\n===== Evaluate BEST model on TEST =====")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    te_loss, te_dice, te_iou, te_acc = eval_one_epoch(
        model, test_loader, criterion, device, epoch=0, phase="Test"
    )
    print(f"Test - loss: {te_loss:.4f}, dice: {te_dice:.4f}, iou: {te_iou:.4f}, acc: {te_acc:.4f}")


if __name__ == "__main__":
    main()
