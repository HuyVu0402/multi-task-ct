import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class LungCTMultiTaskDataset(Dataset):
    """
    Dataset multi-task:
      - input: img hoặc img_roi
      - seg: mask_lesion
      - cls: label_cls (0=normal,1=covid)
    """
    def __init__(self,
                 data_root: str,
                 split_txt: str,
                 meta_csv: str,
                 use_roi: bool = True):
        """
        Args:
            data_root: thư mục gốc processed/covid_normal
            split_txt: path tới train.txt / val.txt / test.txt
            meta_csv: path tới meta.csv
            use_roi: True -> dùng img_roi, False -> dùng img
        """
        self.data_root = data_root
        self.use_roi = use_roi

        # ----- load meta.csv thành dict {slice_id: row_dict} -----
        self.meta = {}
        with open(meta_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ví dụ img = "images/img/p3_z027.png"
                img_rel = row["img"]
                slice_id = os.path.splitext(os.path.basename(img_rel))[0]
                self.meta[slice_id] = row

        # ----- đọc danh sách slice_id từ split file -----
        with open(split_txt, "r") as f:
            self.slice_ids = [line.strip() for line in f if line.strip()]

        # filter: chỉ giữ slice có trong meta
        self.slice_ids = [sid for sid in self.slice_ids if sid in self.meta]
        if len(self.slice_ids) == 0:
            raise ValueError(f"Không có sample nào trong {split_txt} trùng meta.csv")

    def __len__(self):
        return len(self.slice_ids)

    def _read_gray_png(self, rel_path: str):
        abs_path = os.path.join(self.data_root, rel_path)
        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(abs_path)
        return img

    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        row = self.meta[slice_id]

        img_key = "img_roi" if self.use_roi else "img"
        img_rel = row[img_key]
        mask_rel = row["mask_lesion"]
        label_cls = int(row["label_cls"])

        img = self._read_gray_png(img_rel)
        mask = self._read_gray_png(mask_rel)

        # convert 0/255 -> 0/1
        img = img.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) > 0).astype(np.float32)

        # [C,H,W]
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        label_t = torch.tensor(label_cls, dtype=torch.long)

        return {
            "id": slice_id,
            "image": img_t,
            "mask": mask_t,
            "label": label_t,
        }
