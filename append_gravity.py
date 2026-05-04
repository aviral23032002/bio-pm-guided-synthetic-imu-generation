import h5py
import numpy as np
import os
from tqdm import tqdm

TOKEN_STORE = "results_wisdm_v2_6class/token_store.hdf5"
PREPROCESSED_DIR = "preprocessed_wisdm_v2_6class"
OUTPUT_KEY = "gravity_vecs"

def append_gravity():
    if not os.path.exists(TOKEN_STORE):
        print(f"Error: {TOKEN_STORE} not found.")
        return

    with h5py.File(TOKEN_STORE, "a") as f_store:
        # 1. Get Subject IDs to know which files to load
        pids = f_store["subject_ids"][:]
        n_windows = len(pids)
        print(f"Token store has {n_windows} windows.")

        # 2. Prepare gravity array
        # We'll use 900 dimensions (300 samples * 3 axes)
        gravity_array = np.zeros((n_windows, 900), dtype=np.float32)

        unique_pids = np.unique(pids).astype(int)
        
        current_idx = 0
        for pid in tqdm(unique_pids, desc="Processing subjects"):
            # Path to the preprocessed file for this subject
            me_path = os.path.join(PREPROCESSED_DIR, f"Data_MeLabel_{pid}.h5")
            
            if not os.path.exists(me_path):
                print(f"Warning: {me_path} not found. Skipping subject {pid}.")
                continue
                
            with h5py.File(me_path, "r") as f_me:
                # Extract x_gravity (W, 300, 3)
                # Note: BioPM uses x_gravity for the low-pass signal
                g = f_me["x_gravity"][:]
                
                # Flatten to (W, 900)
                g_flat = g.reshape(g.shape[0], -1)
                
                # Find where this subject's windows are in the token store
                mask = (pids == pid)
                count = np.sum(mask)
                
                if count != g_flat.shape[0]:
                    print(f"Warning: Window count mismatch for subject {pid}. "
                          f"Store: {count}, Preprocessed: {g_flat.shape[0]}")
                    # We only take what fits or pad if necessary, 
                    # but usually these align exactly.
                    size = min(count, g_flat.shape[0])
                    gravity_array[mask][:size] = g_flat[:size]
                else:
                    gravity_array[mask] = g_flat

        # 3. Save to HDF5
        if OUTPUT_KEY in f_store:
            print(f"Overwriting existing '{OUTPUT_KEY}'...")
            del f_store[OUTPUT_KEY]
            
        f_store.create_dataset(OUTPUT_KEY, data=gravity_array)
        print(f"Successfully added '{OUTPUT_KEY}' to {TOKEN_STORE}")

if __name__ == "__main__":
    append_gravity()
