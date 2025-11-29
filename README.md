# Multi-task CT Lung Project

Bài tập lớn môn **MAT3563 - Một số vấn đề chọn lọc về thị giác máy tính**.

## Thành viên:
1. Vũ Quang Huy - 22001596
2. Nguyễn Huy Hoàng - 22001587
3. Nguyễn Trí Hiếu - 22001581
4. Lê Tuấn Hiệp - 22001577

## Link tải dữ liệu
1. Covid-19 & Normal CT Segmentation Dataset: https://data.mendeley.com/datasets/pfmgfpwnmm/1
2. COVID‑19 CT Lung and Infection Segmentation Dataset: https://medicalsegmentation.com/covid19/
## Nội dung chính

- Tiền xử lý CT 3D → lát cắt 2D
- Tách phổi bằng pretrained lungmask R231CovidWeb
- Tạo ROI, mask phổi, mask tổn thương và meta.csv
- Huấn luyện:
  - U-Net (segmentation-only) cho tổn thương phổi
  - U-Net Multi-task (segmentation + classification COVID/NORMAL)

## Phần việc:
- Tìm kiếm dữ liệu, trực quan hóa thông tin, tách vùng phổi: Vũ Quang Huy
- Chuyển dữ liệu từ 3D sang 2D, normalize ảnh: Nguyễn Trí Hiếu.
- Viết mô hình U-Net thực hiện phân đoạn với dữ liệu tiền xử lý: Nguyễn Huy Hoàng.
- Viết mô hình U-Net thực hiện đa tác vụ với dữ liệu tiền xử lý: Lê Tuấn Hiệp.

## Cấu trúc thư mục
multi_task_ct/
│
├── checkpoints/                 # Lưu trọng số mô hình đã huấn luyện
│   ├── unet_multitask_best.pth
│   ├── unet_multitask_last.pth
│   ├── unet_seg_best.pth
│   └── unet_seg_last.pth
│
├── data/
│   ├── raw/                     # Dữ liệu thô (.hdr, .img, .nii.gz,...)
│   └── processed/               # Dữ liệu sau tiền xử lý (2D PNG, meta.csv, splits)
│       ├── covid_normal/
│       │   ├── images/
│       │   │   ├── img/
│       │   │   └── img_roi/
│       │   ├── masks/
│       │   │   ├── lesion/
│       │   │   └── lung/
│       │   ├── meta.csv
│       │   └── splits/
│       │       ├── train.txt
│       │       ├── val.txt
│       │       └── test.txt
│
├── logs/
│   ├── unet_multitask_log.csv   # Lịch sử loss/Dice/acc cho U-Net multi-task
│   └── unet_seg_log.csv         # Lịch sử loss/Dice cho U-Net segmentation
│
├── notebooks/                   # Notebook dùng để trực quan hóa & demo
│   ├── 000_setup_all.ipynb
│   ├── 01_covid_lung_infection_info.ipynb
│   ├── 02_covid19_normal.ipynb
│   ├── 03_preprocess_dataset.ipynb
│   └── 05_check_test_model.ipynb
│
├── output/                      # Chứa hình ảnh kết quả (nếu cần xuất)
│
├── src/                         # Mã nguồn chính của dự án
│   ├── data/                    # Dataset loader + tạo meta.csv
│   │   ├── dataset_ct.py
│   │   └── make_splits.py
│   │
│   ├── losses/                  # Hàm loss (Dice, BCE-Dice,…)
│   │   └── losses.py
│   │
│   ├── models/                  # Các mô hình UNet, UNet++
│   │   ├── unet.py
│   │   ├── unet_multitask.py
│   │   └── __init__.py
│   │
│   ├── preprocess/              # Code tiền xử lý
│   │   ├── convert_3d_to_2d.py
│   │   └── lungmask_utils.py
│   │
│   ├── pretrained/              # Chứa pretrained model (lungmask)
│   │   └── lungmask_R231CovidWeb.pth
│   │
│   ├── train/                   # Script training
│   │   ├── train_unet_seg.py
│   │   └── train_unet_multitask.py
│   │
│   └── utils/                   # Hàm tiện ích (metrics, plot,…)
│       ├── metrics.py
│       └── visualization.py
│
├── README.md
└── requirements.txt

## Kịch bản thực nghiệm
- B1: Tải dữ liệu từ 2 link trên.
    + Đổi tên thành bộ đầu tiên thành covid19_normal và bộ thứ hai thành covid19_lung_infection
- B2: trong notebooks/00_setup_all.ipynb:
    + Thay đổi đường dẫn ở cell 1 giải nén dữ liệu: PROJECT_DIR và  INPUT_DIR
    + Chạy cell code 2 để giải nén: !python "{PROJECT_DIR}/src/preprocess/extract_all_data.py" --input_dir "{INPUT_DIR}" --output_dir "{OUTPUT_DIR}"
- B3: Chạy model U-Net và U-Net Multi-task chỉ cần sửa lại đường dẫn đến dữ liệu đã chuẩn hóa 