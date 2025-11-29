"""
covid_normal_convert3d_to_2d.py

- Convert COVID + NORMAL 3D CT -> ảnh 2D (PNG).
- Tách phổi bằng pretrained lungmask (R231CovidWeb).
- Nếu pretrained chưa có thì tự tải về:
      <PROJECT_ROOT>/src/pretrained/lungmask_R231CovidWeb.pth
"""

import os
import csv
import argparse
import urllib.request
import numpy as np
import nibabel as nib
import cv2
from scipy import ndimage as ndi

# thử import lungmask
try:
    from lungmask.mask import LMInferer
    LUNGMASK_AVAILABLE = True
except ImportError:
    LUNGMASK_AVAILABLE = False


HU_MIN = -1000.0
HU_MAX = 400.0
OUTPUT_SIZE = (256, 256)

_LUNG_INFERER = None


# ============================
# Tải pretrained nếu chưa có
# ============================
def ensure_pretrained_exists():
    """
    Đảm bảo pretrained đã tồn tại.
    Nếu chưa có -> tải từ GitHub lungmask.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    pretrained_dir = os.path.join(project_root, "src", "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)

    pretrained_path = os.path.join(pretrained_dir, "lungmask_R231CovidWeb.pth")

    if os.path.exists(pretrained_path):
        return pretrained_path

    url = "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth"
    print(f"\n🌐 Tải pretrained lungmask...\n   URL: {url}")
    print(f"   Lưu vào: {pretrained_path}")

    try:
        urllib.request.urlretrieve(url, pretrained_path)
        print("✅ Tải xong pretrained lungmask.")
    except Exception as e:
        raise RuntimeError(f"❌ Không tải được pretrained lungmask: {e}")

    return pretrained_path


# ============================
# Tạo lungmask inferer
# ============================
def get_lung_inferer():
    global _LUNG_INFERER

    if _LUNG_INFERER is not None:
        return _LUNG_INFERER

    if not LUNGMASK_AVAILABLE:
        raise ImportError(
            "Chưa cài lungmask. Cài bằng:\n"
            "    pip install lungmask SimpleITK"
        )

    pretrained_path = ensure_pretrained_exists()

    print(f"[LUNG] Dùng pretrained: {pretrained_path}")

    inferer = LMInferer(
        modelname="R231CovidWeb",
        modelpath=pretrained_path,
        force_cpu=False,
        batch_size=16,
        volume_postprocessing=True,
        tqdm_disable=True,
    )

    _LUNG_INFERER = inferer
    return inferer


# ============================
# Lung segmentation
# ============================
def segment_lung_3d(ct_hu):
    inferer = get_lung_inferer()

    # lungmask nhận (Z,H,W)
    vol = np.transpose(ct_hu, (2, 0, 1))

    seg = inferer.apply(vol)         # (Z,H,W), giá trị {0,1,2}
    lung = (seg > 0).astype(np.uint8)

    return np.transpose(lung, (1, 2, 0))


# ============================
# tiện ích
# ============================
def get_prefix(name):
    return name.split("_")[0]


def hu_to_uint8(slice_hu):
    x = np.clip(slice_hu, HU_MIN, HU_MAX)
    x = (x - HU_MIN) / (HU_MAX - HU_MIN)
    return (x * 255).astype(np.uint8)


def resize_if_needed(img, is_mask=False):
    target_w, target_h = OUTPUT_SIZE
    h, w = img.shape[:2]
    if (w, h) == (target_w, target_h):
        return img
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def save_png(path, img):
    cv2.imwrite(path, img)


def make_lesion_mask(lbl):
    bg = lbl.min()
    return (np.abs(lbl - bg) > 1e-6).astype(np.uint8)


def make_roi_covid(ct, lung, lesion, dilate_iter=2):
    roi = (lung | lesion).astype(np.uint8)
    struct = ndi.generate_binary_structure(3, 1)
    return ndi.binary_dilation(roi, structure=struct, iterations=dilate_iter).astype(np.uint8)


def make_roi_normal(lung, dilate_iter=1):
    struct = ndi.generate_binary_structure(3, 1)
    return ndi.binary_dilation(lung, structure=struct, iterations=dilate_iter).astype(np.uint8)


# ============================
# Liệt kê case
# ============================
def list_covid_cases(part1_ct, part1_lbl, part2_ct, part2_lbl):
    cases = []

    # ---------------- Part 1 ----------------
    ct_map = {}
    lbl_map = {}

    for f in os.listdir(part1_ct):
        if f.endswith(".hdr"):
            prefix = get_prefix(f.split(".")[0])
            ct_map[prefix] = os.path.join(part1_ct, f)

    for f in os.listdir(part1_lbl):
        if f.endswith(".hdr"):
            prefix = get_prefix(f.split(".")[0])
            lbl_map[prefix] = os.path.join(part1_lbl, f)

    for p in ct_map:
        if p in lbl_map:
            cases.append({
                "case_id": p,
                "part": "Part1",
                "ct_path": ct_map[p],
                "label_path": lbl_map[p],
                "type": "covid",
            })

    # ---------------- Part 2 ----------------
    ct_map = {}
    lbl_map = {}

    for f in os.listdir(part2_ct):
        if f.endswith(".hdr"):
            prefix = get_prefix(f.split(".")[0])
            ct_map[prefix] = os.path.join(part2_ct, f)

    for f in os.listdir(part2_lbl):
        if f.endswith(".hdr"):
            prefix = get_prefix(f.split(".")[0])
            lbl_map[prefix] = os.path.join(part2_lbl, f)

    for p in ct_map:
        if p in lbl_map:
            cases.append({
                "case_id": p,
                "part": "Part2",
                "ct_path": ct_map[p],
                "label_path": lbl_map[p],
                "type": "covid",
            })

    return cases


def list_normal_cases(normal_ct_dir):
    cases = []
    for f in os.listdir(normal_ct_dir):
        if f.endswith(".hdr"):
            prefix = get_prefix(f.split(".")[0])
            cases.append({
                "case_id": prefix,
                "part": "Normal",
                "ct_path": os.path.join(normal_ct_dir, f),
                "label_path": None,
                "type": "normal",
            })
    return cases


# ============================
# Process COVID case
# ============================
def process_covid_case(case, out_dirs, out_root, writer, file_obj):
    out_img, out_roi, out_les, out_lung = out_dirs

    cid = case["case_id"]
    part = case["part"]

    # ===== CHECK IF CASE ALREADY PROCESSED (theo file ảnh) =====
    first_slice = os.path.join(out_img, f"{cid}_z001.png")
    if os.path.exists(first_slice):
        print(f"[SKIP] COVID {cid} đã có file ảnh → bỏ qua (theo ảnh).")
        return
    # ==========================================================

    print(f"\n[COVID][{part}] Case {cid}")

    # Load CT & Label
    ct_nib = nib.load(case["ct_path"])
    lbl_nib = nib.load(case["label_path"])

    ct = ct_nib.get_fdata().astype(np.float32)
    lbl = lbl_nib.get_fdata().astype(np.float32)

    # CHECK SHAPE
    if ct.shape != lbl.shape:
        print(f"[SKIP] COVID {cid} shape CT {ct.shape} != LABEL {lbl.shape} → bỏ qua case này.")
        return

    H, W, Z = ct.shape

    lung = segment_lung_3d(ct)
    lesion = make_lesion_mask(lbl)
    roi = make_roi_covid(ct, lung, lesion)

    slice_count = 0

    for z in range(Z):
        if lung[:, :, z].sum() == 0 and lesion[:, :, z].sum() == 0:
            continue

        name = f"{cid}_z{z+1:03d}"

        img_u8 = hu_to_uint8(ct[:, :, z])
        roi_u8 = hu_to_uint8(np.where(roi[:, :, z], ct[:, :, z], HU_MIN))

        m_lung = (lung[:, :, z] * 255).astype(np.uint8)
        m_les = (lesion[:, :, z] * 255).astype(np.uint8)

        img_u8 = resize_if_needed(img_u8)
        roi_u8 = resize_if_needed(roi_u8)
        m_lung = resize_if_needed(m_lung, True)
        m_les = resize_if_needed(m_les, True)

        p_img = os.path.join(out_img, name + ".png")
        p_roi = os.path.join(out_roi, name + ".png")
        p_les = os.path.join(out_les, name + ".png")
        p_lung = os.path.join(out_lung, name + ".png")

        save_png(p_img, img_u8)
        save_png(p_roi, roi_u8)
        save_png(p_les, m_les)
        save_png(p_lung, m_lung)

        writer.writerow([
            cid, part, "covid", z, z+1,
            os.path.relpath(p_img, out_root),
            os.path.relpath(p_roi, out_root),
            os.path.relpath(p_les, out_root),
            os.path.relpath(p_lung, out_root),
            int(m_les.sum() > 0),
            1
        ])
        slice_count += 1

    # 🔹 flush sau mỗi case để CSV luôn được cập nhật
    file_obj.flush()
    print(f"  → Ghi xong {slice_count} slice cho case {cid}, đã flush meta.csv.")


# ============================
# Process NORMAL case
# ============================
def process_normal_case(case, out_dirs, out_root, writer, file_obj):
    out_img, out_roi, out_les, out_lung = out_dirs

    cid = case["case_id"]

    # ===== CHECK IF CASE ALREADY PROCESSED (theo file ảnh) =====
    first_slice = os.path.join(out_img, f"{cid}_z001.png")
    if os.path.exists(first_slice):
        print(f"[SKIP] NORMAL {cid} đã có file ảnh → bỏ qua (theo ảnh).")
        return
    # ===========================================================

    print(f"\n[NORMAL] Case {cid}")

    ct_nib = nib.load(case["ct_path"])
    ct = ct_nib.get_fdata().astype(np.float32)

    H, W, Z = ct.shape

    lung = segment_lung_3d(ct)
    roi = make_roi_normal(lung)
    lesion = np.zeros_like(lung)

    slice_count = 0

    for z in range(Z):
        if lung[:, :, z].sum() == 0:
            continue

        name = f"{cid}_z{z+1:03d}"

        img_u8 = hu_to_uint8(ct[:, :, z])
        roi_u8 = hu_to_uint8(np.where(roi[:, :, z], ct[:, :, z], HU_MIN))

        m_lung = (lung[:, :, z] * 255).astype(np.uint8)
        m_les = np.zeros_like(m_lung)

        img_u8 = resize_if_needed(img_u8)
        roi_u8 = resize_if_needed(roi_u8)
        m_lung = resize_if_needed(m_lung, True)
        m_les = resize_if_needed(m_les, True)

        p_img = os.path.join(out_img, name + ".png")
        p_roi = os.path.join(out_roi, name + ".png")
        p_les = os.path.join(out_les, name + ".png")
        p_lung = os.path.join(out_lung, name + ".png")

        save_png(p_img, img_u8)
        save_png(p_roi, roi_u8)
        save_png(p_les, m_les)
        save_png(p_lung, m_lung)

        writer.writerow([
            cid, "Normal", "normal", z, z+1,
            os.path.relpath(p_img, out_root),
            os.path.relpath(p_roi, out_root),
            os.path.relpath(p_les, out_root),
            os.path.relpath(p_lung, out_root),
            0,
            0
        ])
        slice_count += 1

    # 🔹 flush sau mỗi case để CSV luôn được cập nhật
    file_obj.flush()
    print(f"  → Ghi xong {slice_count} slice cho case {cid}, đã flush meta.csv.")


# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True,
                        help="Thư mục gốc chứa folder 'COVID-19 & Normal CT Segmentation Dataset'")
    parser.add_argument("--out_root", required=True,
                        help="Thư mục output (sẽ chứa images/, masks/, meta.csv)")
    args = parser.parse_args()

    raw_root = args.raw_root
    out_root = args.out_root

    # Thư mục chứa 2 folder Organized ...
    dataset_root = os.path.join(raw_root, "COVID-19 & Normal CT Segmentation Dataset")

    # COVID cases
    covid_root = os.path.join(dataset_root, "Organized COVID19 CT Data_rev2")
    part1_ct = os.path.join(covid_root, "Part 1")
    part1_lbl = os.path.join(part1_ct, "CT Labels")
    part2_ct = os.path.join(covid_root, "Part 2")
    part2_lbl = os.path.join(part2_ct, "CT Labels")

    # NORMAL cases
    normal_root = os.path.join(dataset_root, "Organized Normal CT Data")

    # check tồn tại cho chắc
    for p in [dataset_root, covid_root, part1_ct, part1_lbl, part2_ct, part2_lbl, normal_root]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {p}")

    out_img = os.path.join(out_root, "images", "img")
    out_roi = os.path.join(out_root, "images", "img_roi")
    out_les = os.path.join(out_root, "masks", "lesion")
    out_lung = os.path.join(out_root, "masks", "lung")

    for d in [out_img, out_roi, out_les, out_lung]:
        os.makedirs(d, exist_ok=True)

    meta_csv = os.path.join(out_root, "meta.csv")

    # Liệt kê case
    covid_cases = list_covid_cases(part1_ct, part1_lbl, part2_ct, part2_lbl)
    normal_cases = list_normal_cases(normal_root)

    print("Tổng số case COVID :", len(covid_cases))
    print("Tổng số case NORMAL:", len(normal_cases))

    out_dirs = (out_img, out_roi, out_les, out_lung)

    # ===== ĐỌC meta.csv NẾU ĐÃ CÓ: LẤY DANH SÁCH case_id ĐÃ XỬ LÝ =====
    done_case_ids = set()
    file_exists = os.path.exists(meta_csv)
    if file_exists:
        print(f"\n📄 Đã tồn tại meta.csv, đọc để lấy danh sách case_id đã xử lý: {meta_csv}")
        with open(meta_csv, "r", newline="") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                cid = row.get("case_id")
                if cid:
                    done_case_ids.add(cid)
        print(f"🔁 Đã có {len(done_case_ids)} case_id trong meta.csv → sẽ SKIP những case này.\n")
    else:
        print("\n🆕 Chưa có meta.csv, sẽ tạo mới.\n")
    # ==================================================================

    # Mở meta.csv: nếu đã tồn tại thì append, nếu chưa thì tạo mới + ghi header
    with open(meta_csv, "a" if file_exists else "w", newline="", buffering=1) as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "case_id", "part", "type", "z_index", "slice_idx1",
                "img", "img_roi", "mask_lesion", "mask_lung",
                "has_lesion", "label_cls"
            ])
            f.flush()
            print("📝 Tạo mới meta.csv và ghi header, đã flush.")

        # COVID
        for case in covid_cases:
            if case["case_id"] in done_case_ids:
                print(f"[SKIP] COVID {case['case_id']} đã có trong meta.csv → bỏ qua (theo CSV).")
                continue
            process_covid_case(case, out_dirs, out_root, writer, f)

        # NORMAL
        for case in normal_cases:
            if case["case_id"] in done_case_ids:
                print(f"[SKIP] NORMAL {case['case_id']} đã có trong meta.csv → bỏ qua (theo CSV).")
                continue
            process_normal_case(case, out_dirs, out_root, writer, f)

    print("\n🎉 Convert 3D → 2D hoàn thành!")


if __name__ == "__main__":
    main()
