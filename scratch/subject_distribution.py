
import h5py
import numpy as np
import pandas as pd

ACTIVITY_NAMES = {
    0: "Walking", 1: "Jogging", 2: "Upstairs", 
    3: "Downstairs", 4: "Sitting", 5: "Standing"
}

def check_subject_class_distribution(path, title):
    print(f"\n{'='*60}")
    print(f"Subject-wise Class Distribution: {title}")
    print(f"{'='*60}")
    
    with h5py.File(path, "r") as f:
        labels = f["labels"][:].astype(int)
        subjects = f["subject_ids"][:].astype(int)
        
    df = pd.DataFrame({'Subject': subjects, 'Class': labels})
    
    # Pivot table to get counts per subject per class
    dist = df.groupby(['Subject', 'Class']).size().unstack(fill_value=0)
    
    # Rename columns to activity names
    dist.columns = [ACTIVITY_NAMES.get(c, f"C{c}") for c in dist.columns]
    
    # Add a total column
    dist['Total'] = dist.sum(axis=1)
    
    print(dist.to_string())
    
    print(f"\nUnique Subjects: {len(dist)}")
    print(f"Total Windows: {dist['Total'].sum()}")

check_subject_class_distribution("results_wisdm_v2_6class/token_store.hdf5", "WISDM v2 6-Class")
