"""
ct3d_to_2d.py

Chuyển toàn bộ CT 3D + các loại mask (lung, infection, multi)
sang dữ liệu 2D PNG + meta.csv.

Theo chuẩn:
  - images
  - masks_lung
  - masks_infection
  - masks_multi (0/1/2)
"""

import os
import glob
import numpy as np
import nibabel as nib
import cv2
import pandas as pd


# ============================================================
# 1. HÀM CƠ BẢN
# ============================================================

def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)

def normalize_hu(ct, hu_min=-1000, hu_max=400):
    ct = ct.astype(np.float32)
    ct = np.clip(ct, hu_min, hu_max)
    return (ct - hu_min) / (hu_max - hu_min)


def map_mask_multi(mask):
    """
    Mapping:
      0 → 0 (background)
      1 → lung-left  -> 1
      2 → lung-right -> 1
      3 → infection  -> 2
    """
    mask = mask.astype(np.int16)
    out = np.zeros_like(mask, dtype=np.uint8)
    out[(mask == 1) | (mask == 2)] = 1
    out[mask == 3] = 2
    return out


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 2. XỬ LÝ 1 CT VOLUME → NHIỀU SLICE 2D
# ============================================================

def process_one_case(
    case_id,
    ct_path,
    lung_path,
    inf_path,
    multi_path,
    out_img_dir,
    out_lung_dir,
    out_inf_dir,
    out_multi_dir,
    target_size=(512, 512),
    project_dir=None
):
    print(f"🧾 Case: {case_id}")

    ct_vol = load_nifti(ct_path)
    lung_vol = load_nifti(lung_path)
    inf_vol = load_nifti(inf_path)
    multi_vol_raw = load_nifti(multi_path)
    multi_vol = map_mask_multi(multi_vol_raw)

    H, W, Z = ct_vol.shape
    meta_rows = []

    for z in range(Z):
        mask_multi_slice = multi_vol[:, :, z]

        # bỏ slice không có lung/infection
        if np.all(mask_multi_slice == 0):
            continue

        ct_slice = ct_vol[:, :, z]
        lung_slice = lung_vol[:, :, z]
        inf_slice = inf_vol[:, :, z]
        multi_slice = mask_multi_slice

        # normalize HU
        ct_norm = normalize_hu(ct_slice)

        # resize
        ct_resized = cv2.resize(ct_norm, target_size, interpolation=cv2.INTER_LINEAR)
        lung_resized = cv2.resize(lung_slice, target_size, interpolation=cv2.INTER_NEAREST)
        inf_resized = cv2.resize(inf_slice, target_size, interpolation=cv2.INTER_NEAREST)
        multi_resized = cv2.resize(multi_slice, target_size, interpolation=cv2.INTER_NEAREST)

        # file output name
        slice_id = f"{case_id}_z{z:03d}"

        img_path  = os.path.join(out_img_dir,  f"{slice_id}.png")
        lung_path2 = os.path.join(out_lung_dir, f"{slice_id}.png")
        inf_path2 = os.path.join(out_inf_dir,  f"{slice_id}.png")
        multi_path2 = os.path.join(out_multi_dir, f"{slice_id}.png")

        # save images
        cv2.imwrite(img_path, (ct_resized * 255).astype(np.uint8))
        cv2.imwrite(lung_path2, lung_resized.astype(np.uint8))
        cv2.imwrite(inf_path2, inf_resized.astype(np.uint8))
        cv2.imwrite(multi_path2, multi_resized.astype(np.uint8))

        # infection label
        has_inf = int(np.any(multi_resized == 2))

        # relative path for meta CSV
        if project_dir:
            img_rel = os.path.relpath(img_path, project_dir).replace("\\", "/")
            lung_rel = os.path.relpath(lung_path2, project_dir).replace("\\", "/")
            inf_rel = os.path.relpath(inf_path2, project_dir).replace("\\", "/")
            multi_rel = os.path.relpath(multi_path2, project_dir).replace("\\", "/")
        else:
            img_rel, lung_rel, inf_rel, multi_rel = (
                img_path, lung_path2, inf_path2, multi_path2
            )

        meta_rows.append({
            "case_id": case_id,
            "slice_idx": z,
            "slice_id": slice_id,
            "img_path": img_rel,
            "mask_lung_path": lung_rel,
            "mask_infection_path": inf_rel,
            "mask_multi_path": multi_rel,
            "has_infection": has_inf
        })

    return meta_rows


# ============================================================
# 3. DUYỆT TOÀN BỘ DATASET
# ============================================================

def process_covid_lung_infection(
    raw_root,
    out_root,
    target_size=(512, 512),
    project_dir=None
):
    """
    raw_root:
        data/raw/covid_lung_infection

    out_root:
        data/processed/covid_lung_infection
    """

    # input
    CT_DIR = os.path.join(raw_root, "CT")
    LUNG_DIR = os.path.join(raw_root, "mask_lung")
    INF_DIR = os.path.join(raw_root, "mask_infection")
    MULTI_DIR = os.path.join(raw_root, "mask_multi")

    # output
    out_img_dir  = os.path.join(out_root, "images")
    out_lung_dir = os.path.join(out_root, "masks_lung")
    out_inf_dir  = os.path.join(out_root, "masks_infection")
    out_multi_dir = os.path.join(out_root, "masks_multi")
    out_meta_csv = os.path.join(out_root, "meta.csv")

    ensure_dir(out_img_dir)
    ensure_dir(out_lung_dir)
    ensure_dir(out_inf_dir)
    ensure_dir(out_multi_dir)

    ct_files = sorted(glob.glob(os.path.join(CT_DIR, "*.nii")))
    meta_all = []

    print("====================================")
    print("   BẮT ĐẦU CHUYỂN 3D → 2D")
    print("====================================")
    print(f"Tìm thấy {len(ct_files)} CT volume\n")

    for ct_path in ct_files:
        case_id = os.path.splitext(os.path.basename(ct_path))[0]

        lung_path = os.path.join(LUNG_DIR, case_id + ".nii")
        inf_path = os.path.join(INF_DIR, case_id + ".nii")
        multi_path = os.path.join(MULTI_DIR, case_id + ".nii")

        if not (os.path.exists(lung_path) and os.path.exists(inf_path) and os.path.exists(multi_path)):
            print(f"⚠️  Bỏ qua {case_id}: thiếu mask")
            continue

        meta_rows = process_one_case(
            case_id, ct_path, lung_path, inf_path, multi_path,
            out_img_dir, out_lung_dir, out_inf_dir, out_multi_dir,
            target_size=target_size,
            project_dir=project_dir
        )
        meta_all.extend(meta_rows)

    df = pd.DataFrame(meta_all)
    df.to_csv(out_meta_csv, index=False)

    print(f"\n✔ DONE. Tổng số slice: {len(df)}")
    print(f"CSV lưu tại: {out_meta_csv}")

    return df
