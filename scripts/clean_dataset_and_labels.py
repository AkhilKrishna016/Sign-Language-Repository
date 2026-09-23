import os
import shutil
import json

def clean_dataset_and_rebuild_labels():
    train_dir = os.path.join("dataset", "character dataset", "asl_alphabet_train", "asl_alphabet_train")
    test_dir = os.path.join("dataset", "character dataset", "asl_alphabet_test", "asl_alphabet_test")

    # 1. Clean train non-character folders
    removed_train = []
    for folder in ["del", "nothing", "space"]:
        p = os.path.join(train_dir, folder)
        if os.path.exists(p):
            shutil.rmtree(p)
            removed_train.append(folder)

    # 2. Clean test non-character files
    removed_test = []
    for fname in ["nothing_test.jpg", "space_test.jpg"]:
        p = os.path.join(test_dir, fname)
        if os.path.exists(p):
            os.remove(p)
            removed_test.append(fname)

    print(f"Removed train folders: {removed_train}")
    print(f"Removed test files: {removed_test}")

    # 3. List remaining classes & counts
    remaining_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_counts = {}
    for c in remaining_classes:
        cp = os.path.join(train_dir, c)
        class_counts[c] = len(os.listdir(cp))

    print(f"\nTotal remaining classes: {len(remaining_classes)}")
    print("Per-class image counts:")
    for c, cnt in class_counts.items():
        print(f"  {c}: {cnt} images")

    # Check minimum threshold
    min_threshold = 50
    flagged_low = [c for c, cnt in class_counts.items() if cnt < min_threshold]
    if flagged_low:
        print(f"\nClasses with fewer than {min_threshold} images: {flagged_low}")
    else:
        print(f"\nClasses with fewer than {min_threshold} images: None! All 26 classes meet threshold.")

    # 4. Save clean label map
    label_map = {
        "classes": remaining_classes,
        "class_to_idx": {c: i for i, c in enumerate(remaining_classes)},
        "idx_to_class": {i: c for i, c in enumerate(remaining_classes)},
        "num_classes": len(remaining_classes)
    }

    os.makedirs("backend/models", exist_ok=True)
    with open("backend/models/character_label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\nSaved clean character label map to backend/models/character_label_map.json")
    return label_map, class_counts

if __name__ == "__main__":
    clean_dataset_and_rebuild_labels()
