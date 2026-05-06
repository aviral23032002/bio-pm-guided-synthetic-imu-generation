
import h5py
import numpy as np
from car_imu_decoder import ACTIVITY_NAMES, NUM_CLASSES

with h5py.File("results_wisdm_v2_6class/token_store.hdf5", "r") as f:
    labels = f["labels"][:].astype(int)

total = len(labels)
print(f"Total windows: {total}")
print(f"\nClass distribution:")
for cls_id in range(NUM_CLASSES):
    n   = (labels == cls_id).sum()
    pct = 100 * n / total
    print(f"  {ACTIVITY_NAMES[cls_id]:<12} {n:>5} ({pct:.1f}%)")
