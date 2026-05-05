import h5py
import numpy as np

# ── Setup ─────────────────────────────────────────────────────────────────────
REAL_TOKENS = "results_wisdm_v2_6class/token_store.hdf5"
SYN_TOKENS  = "synthetic_tokens_v2_6class_5pct/synthetic_tokens.hdf5"

ACTIVITY_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]

def check_positional_similarity(real_path, syn_path):
    print(f"Loading Real: {real_path}")
    with h5py.File(real_path, 'r') as fr:
        real_tokens = fr['tokens'][:].astype(np.float32)
        real_labels = fr['labels'][:].astype(int)
    
    print(f"Loading Syn:  {syn_path}")
    with h5py.File(syn_path, 'r') as fs:
        syn_tokens = fs['tokens'][:].astype(np.float32)
        syn_labels = fs['labels'][:].astype(int)

    print("\nPer-Position Cosine Similarity (Real vs Synthetic Mean Profiles):")
    print("-" * 60)
    
    for cls_id in range(6):
        real_mask = real_labels == cls_id
        syn_mask  = syn_labels  == cls_id

        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            print(f"{ACTIVITY_NAMES[cls_id]}: Missing data")
            continue

        # Compute per-position mean: (192, 64)
        real_pos_mean = real_tokens[real_mask].mean(axis=0)
        syn_pos_mean  = syn_tokens[syn_mask].mean(axis=0)

        # Per-position cosine similarity
        cos_sims = []
        for t in range(192):
            r_vec = real_pos_mean[t]
            s_vec = syn_pos_mean[t]
            sim = np.dot(r_vec, s_vec) / (np.linalg.norm(r_vec) * np.linalg.norm(s_vec) + 1e-8)
            cos_sims.append(sim)
        
        cos_sims = np.array(cos_sims)

        print(f"{ACTIVITY_NAMES[cls_id]:<12}:")
        print(f"  Mean pos cos_sim: {cos_sims.mean():.4f}")
        print(f"  Min  pos cos_sim: {cos_sims.min():.4f} at position {cos_sims.argmin()}")
        print(f"  Max  pos cos_sim: {cos_sims.max():.4f} at position {cos_sims.argmax()}")
        print(f"  Positions < 0.8:  {(cos_sims < 0.8).sum()}")
        print(f"  Positions < 0.6:  {(cos_sims < 0.6).sum()}")

if __name__ == "__main__":
    check_positional_similarity(REAL_TOKENS, SYN_TOKENS)
