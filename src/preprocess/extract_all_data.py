import argparse
import os
import zipfile
import gzip
import shutil
import glob


# ==============================
# 1. ZIP: từ INPUT -> OUTPUT (KHÔNG XOÁ INPUT)
# ==============================
def copy_and_extract_zips_from_input(input_dir: str, output_dir: str):
    print(f"📁 Quét .zip trong INPUT (chỉ đọc, không xóa): {input_dir}")

    zip_files = glob.glob(os.path.join(input_dir, "**", "*.zip"), recursive=True)
    print(f"🔍 Tìm thấy {len(zip_files)} file .zip trong INPUT_DIR")

    for zip_path in zip_files:
        root = os.path.dirname(zip_path)
        rel_root = os.path.relpath(root, input_dir)

        target_dir = os.path.join(
            output_dir,
            rel_root,
            os.path.splitext(os.path.basename(zip_path))[0]
        )
        os.makedirs(target_dir, exist_ok=True)

        print(f"\n📦 Giải nén (INPUT -> OUTPUT): {zip_path}")
        print(f"   ➜ Vào: {target_dir}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
        except Exception as e:
            print(f"   ❌ Lỗi khi giải nén {zip_path}: {e}")

    print("\n✅ Hoàn thành bước giải nén .zip từ INPUT sang OUTPUT.\n")


# ==============================
# 2. ZIP: xử lý zip lồng zip trong OUTPUT (CÓ XOÁ)
# ==============================
def extract_zip_files_recursive_in_output(output_dir: str):
    iteration = 1
    while True:
        zip_files = glob.glob(os.path.join(output_dir, "**", "*.zip"), recursive=True)
        if not zip_files:
            print(f"\n✅ Không còn file .zip nào trong OUTPUT: {output_dir}")
            break

        print(f"\n🔁 Vòng {iteration}: tìm thấy {len(zip_files)} file .zip trong OUTPUT")

        for zip_path in zip_files:
            root = os.path.dirname(zip_path)
            target_dir = os.path.join(
                root,
                os.path.splitext(os.path.basename(zip_path))[0]
            )
            os.makedirs(target_dir, exist_ok=True)

            print(f"\n📦 Giải nén (OUTPUT): {zip_path}")
            print(f"   ➜ Vào: {target_dir}")

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(target_dir)
                print("   ✅ Giải nén xong, xóa file .zip (OUTPUT)")
                os.remove(zip_path)
            except Exception as e:
                print(f"   ❌ Lỗi khi giải nén {zip_path}: {e}")
                print("   ⚠️ Giữ lại file .zip để kiểm tra sau")

        iteration += 1

    print(f"\n✅ Hoàn thành xử lý zip lồng zip trong OUTPUT: {output_dir}\n")


# ==============================
# 3. NII.GZ: từ INPUT -> OUTPUT (KHÔNG XOÁ INPUT)
# ==============================
def convert_nii_gz_from_input_to_output(input_dir: str, output_dir: str):
    print(f"📁 Quét .nii.gz trong INPUT (chỉ đọc, không xóa): {input_dir}")

    nii_gz_files = glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True)
    print(f"🔍 Tìm thấy {len(nii_gz_files)} file .nii.gz trong INPUT_DIR")

    for gz_path in nii_gz_files:
        root = os.path.dirname(gz_path)
        rel_root = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel_root)
        os.makedirs(out_root, exist_ok=True)

        nii_name = os.path.basename(gz_path)[:-3]
        nii_path = os.path.join(out_root, nii_name)

        if os.path.exists(nii_path):
            print(f"   ⚠️ Đã tồn tại (bỏ qua): {nii_path}")
            continue

        print(f"\n🩻 Giải nén NIfTI (INPUT -> OUTPUT): {gz_path}")
        print(f"   ➜ Vào: {nii_path}")

        try:
            with gzip.open(gz_path, "rb") as f_in, open(nii_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"   ❌ Lỗi khi giải nén {gz_path}: {e}")

    print("\n✅ Hoàn thành chuyển .nii.gz từ INPUT sang OUTPUT.\n")


# ==============================
# 4. NII.GZ: xử lý trong OUTPUT (CÓ XOÁ)
# ==============================
def convert_nii_gz_in_output(output_dir: str):
    print(f"📁 Quét .nii.gz trong OUTPUT (sẽ xoá sau khi giải): {output_dir}")

    nii_gz_files = glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True)
    print(f"🔍 Tìm thấy {len(nii_gz_files)} file .nii.gz trong OUTPUT_DIR")

    for gz_path in nii_gz_files:
        root = os.path.dirname(gz_path)
        nii_name = os.path.basename(gz_path)[:-3]
        nii_path = os.path.join(root, nii_name)

        print(f"\n🩻 Giải nén NIfTI (OUTPUT): {gz_path}")
        print(f"   ➜ Vào: {nii_path}")

        if os.path.exists(nii_path):
            print("   ⚠️ File .nii đã tồn tại, bỏ qua.")
            continue

        try:
            with gzip.open(gz_path, "rb") as f_in, open(nii_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print("   ✅ Giải nén xong, xóa file .nii.gz (OUTPUT)")
            os.remove(gz_path)
        except Exception as e:
            print(f"   ❌ Lỗi khi giải nén {gz_path}: {e}")
            print("   ⚠️ Giữ lại file .nii.gz để kiểm tra sau")

    print("\n✅ Hoàn thành xử lý .nii.gz trong OUTPUT.\n")


# ==============================
# 5. Gộp folder trùng tên
# ==============================
def fix_duplicate_subfolders(root_dir: str):
    print(f"\n🧹 Đang xử lý gộp các folder trùng tên bên trong: {root_dir}")

    for current_root, dirnames, _ in os.walk(root_dir, topdown=True):
        base = os.path.basename(current_root)
        for d in list(dirnames):
            if d == base:
                inner_dir = os.path.join(current_root, d)
                print(f"\n🔁 Phát hiện folder lồng nhau: {current_root} / {d}")
                print(f"   ➜ Gộp {inner_dir} lên {current_root}")

                for item in os.listdir(inner_dir):
                    src = os.path.join(inner_dir, item)
                    dst = os.path.join(current_root, item)

                    if os.path.exists(dst):
                        print(f"   ⚠️ Đã tồn tại: {dst} -> bỏ qua move {src}")
                        continue

                    try:
                        shutil.move(src, dst)
                    except Exception as e:
                        print(f"   ❌ Lỗi khi move {src} -> {dst}: {e}")

                try:
                    os.rmdir(inner_dir)
                    print(f"   ✅ Đã xóa folder con: {inner_dir}")
                except OSError as e:
                    print(f"   ⚠️ Không xóa được {inner_dir}: {e}")

                dirnames.remove(d)

    print("\n✅ Hoàn thành bước fix folder trùng tên.\n")


# ==============================
# 6. Summary
# ==============================
def summary_after_process(root_dir: str):
    zip_files = glob.glob(os.path.join(root_dir, "**", "*.zip"), recursive=True)
    nii_gz_files = glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True)
    nii_files = glob.glob(os.path.join(root_dir, "**", "*.nii"), recursive=True)

    print("\n========== SUMMARY TRONG OUTPUT ==========")
    print(f"📦 ZIP còn lại     : {len(zip_files)}")
    print(f"🩻 NII.GZ còn lại  : {len(nii_gz_files)}")
    print(f"🩻 NII (đã tạo)    : {len(nii_files)}")
    print("===========================================")


# ==============================
# 7. MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("==========================================")
    print("  🚀 BẮT ĐẦU CHUẨN BỊ DỮ LIỆU NIFTI")
    print("==========================================")
    print(f"📥 INPUT  (read-only): {input_dir}")
    print(f"📤 OUTPUT (working)  : {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    copy_and_extract_zips_from_input(input_dir, output_dir)
    convert_nii_gz_from_input_to_output(input_dir, output_dir)

    extract_zip_files_recursive_in_output(output_dir)
    convert_nii_gz_in_output(output_dir)

    fix_duplicate_subfolders(output_dir)
    summary_after_process(output_dir)

    print("\n🎉 DONE! INPUT giữ nguyên, OUTPUT đã được giải nén & dọn sạch zip/nii.gz.\n")


if __name__ == "__main__":
    main()