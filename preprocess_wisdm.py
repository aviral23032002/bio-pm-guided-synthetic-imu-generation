import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    file_path = 'WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    
    print("1. Loading and Cleaning Data...")
    # The file has no header and sometimes contains bad lines; we skip them
    columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.read_csv(file_path, header=None, names=columns, on_bad_lines='skip')
    
    # WISDM has a quirk where the 'z' column ends with a semicolon. 
    # We need to strip it out and convert to float.
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
    df.dropna(inplace=True)
    
    print(f"Total raw samples loaded: {len(df)}")

    # ---------------------------------------------------------
    print("\n2. Class Balance Audit...")
    class_counts = df['activity'].value_counts()
    class_percentages = (class_counts / len(df)) * 100
    
    print("Class Distribution:")
    for activity, pct in class_percentages.items():
        print(f" - {activity}: {pct:.2f}%")
        
    minority_classes = class_percentages[class_percentages < 10.0].index.tolist()
    print(f"\n[FLAG] Minority classes (<10%): {minority_classes}")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("WISDM Activity Distribution")
    plt.ylabel("Samples")
    plt.savefig("class_distribution.png")
    print("Saved distribution plot to 'class_distribution.png'")

    # ---------------------------------------------------------
    print("\n3. Building Windowing Pipeline...")
    window_size = 200  # 10 seconds at 20Hz
    step_size = 100    # 50% overlap
    
    windows = []
    labels = []
    subject_ids = []

    # Group by user and activity so windows don't cross boundaries
    for (user, activity), group in df.groupby(['user', 'activity']):
        data = group[['x', 'y', 'z']].values
        
        # Calculate how many full windows we can extract
        num_windows = (len(data) - window_size) // step_size + 1
        
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            
            windows.append(data[start:end])
            labels.append(activity)
            subject_ids.append(user)

    # Convert to numpy arrays
    windows = np.array(windows)
    # HDF5 strings need to be ASCII encoded
    labels = np.array(labels, dtype='S') 
    subject_ids = np.array(subject_ids)
    
    print(f"Final Tensor Shape: {windows.shape}") # Should be (N, 200, 3)
    print(f"Total extracted windows: {len(windows)}")

    # ---------------------------------------------------------
    print("\n4. Saving to HDF5...")
    output_file = 'wisdm_windows.hdf5'
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('windows', data=windows)
        f.create_dataset('labels', data=labels)
        f.create_dataset('subject_ids', data=subject_ids)
        
    print(f"Success! Data saved to '{output_file}' ready for Bio-PM.")

if __name__ == "__main__":
    main()