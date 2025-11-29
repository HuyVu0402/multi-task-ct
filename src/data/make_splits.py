import os
import csv
import argparse
import random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = args.data_root
    meta_csv = os.path.join(data_root, "meta.csv")
    splits_dir = os.path.join(data_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # 1) Gom slice theo case_id
    case_to_slices = defaultdict(list)
    with open(meta_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"]      # từ meta.csv
            img_rel = row["img"]
            sid = os.path.splitext(os.path.basename(img_rel))[0]  # vd p49_z006
            case_to_slices[case_id].append(sid)

    case_ids = list(case_to_slices.keys())
    print(f"Tổng số case: {len(case_ids)}")

    random.seed(args.seed)
    random.shuffle(case_ids)

    n = len(case_ids)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_cases = case_ids[:n_train]
    val_cases   = case_ids[n_train:n_train + n_val]
    test_cases  = case_ids[n_train + n_val:]

    def expand_cases(cases):
        sids = []
        for cid in cases:
            sids.extend(case_to_slices[cid])
        return sids

    train_ids = expand_cases(train_cases)
    val_ids   = expand_cases(val_cases)
    test_ids  = expand_cases(test_cases)

    print(f"Train case: {len(train_cases)}, slices: {len(train_ids)}")
    print(f"Val   case: {len(val_cases)}, slices: {len(val_ids)}")
    print(f"Test  case: {len(test_cases)}, slices: {len(test_ids)}")

    def write_list(path, ids):
        with open(path, "w") as f:
            for sid in ids:
                f.write(sid + "\n")

    write_list(os.path.join(splits_dir, "train.txt"), train_ids)
    write_list(os.path.join(splits_dir, "val.txt"), train_ids)
    write_list(os.path.join(splits_dir, "test.txt"), test_ids)

    print("Đã lưu splits_by_case vào:", splits_dir)

if __name__ == "__main__":
    main()
