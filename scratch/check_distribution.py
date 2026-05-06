
import h5py
import numpy as np
import os

ACTIVITY_NAMES = {
    0: "Walking", 1: "Jogging", 2: "Upstairs", 
    3: "Downstairs", 4: "Sitting", 5: "Standing"
}

def check_hdf5(name, path):
    if not os.path.exists(path):
        print(f"\n{name} ({path}): NOT FOUND")
        return
    print(f"\n{name} ({path}):")
    with h5py.File(path, "r") as f:
        labels = f["labels"][:].astype(int)
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        for cls_id, count in zip(unique, counts):
            name_cls = ACTIVITY_NAMES.get(cls_id, f"Unknown({cls_id})")
            pct = 100 * count / total
            print(f"  {name_cls:<12}: {count:>5} ({pct:.1f}%)")
        print(f"  {'TOTAL':<12}: {total:>5}")

print("Checking WISDM Class Distributions...")
check_hdf5("WISDM v1 (Week 1)", "results_week1/token_store.hdf5")
check_hdf5("WISDM v2 Clean", "results_wisdm_v2_clean/token_store.hdf5")
check_hdf5("WISDM v2 6-Class", "results_wisdm_v2_6class/token_store.hdf5")
