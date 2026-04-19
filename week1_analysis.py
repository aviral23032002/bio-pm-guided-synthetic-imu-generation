#!/usr/bin/env python3
"""
week1_analysis.py — Week 1 Days 3-5 analysis for CAR-IMU project.

Covers (in order):
  1. Token extraction  → token_store.hdf5
                         (full Bio-PM token matrices Z ∈ R^(N×D) per window)
  2. NN sanity check   → 5 nearest neighbours in token space, same-class check
  3. UMAP              → activity-coloured UMAP of mean-pooled tokens
  4. Subject embeddings → subject_embeddings.npy  (mean-pooled per subject)
  5. UMAP              → subject-coloured UMAP of subject-style embeddings
  6. Baseline HAR      → linear probe + MLP, LOSO macro-F1 (real-only baseline)

Usage (run AFTER extract_features on all subjects):
    python week1_analysis.py \
        --data_dir    preprocessed_biopm \
        --checkpoint  CS690TR/checkpoints/checkpoint.pt \
        --features    features/biopm_features_all.npz \
        --out_dir     results_week1

If you only have the single-subject test run, pass:
    --data_dir  preprocessed_biopm_test
    --features  features/biopm_features.npz
"""

import os
import sys
import argparse
import warnings
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# ── Bio-PM imports ─────────────────────────────────────────────────────────────
BIOPM_DIR = os.path.join(os.path.dirname(__file__), "CS690TR")
sys.path.insert(0, BIOPM_DIR)

from src.models.biopm import load_pretrained_encoder, masked_mean_std
from src.data.dataset import MovementElementDataset
from src.data.preprocessing import load_preprocessed_h5

# ── Activity label mapping ─────────────────────────────────────────────────────
ACTIVITY_NAMES = {
    0: "Walking",
    1: "Jogging",
    2: "Upstairs",
    3: "Downstairs",
    4: "Sitting",
    5: "Standing",
}
ACTIVITY_COLORS = {
    0: "#4361ee",   # blue
    1: "#f72585",   # pink
    2: "#7209b7",   # purple
    3: "#3a86ff",   # light blue
    4: "#fb8500",   # orange
    5: "#06d6a0",   # teal
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TOKEN EXTRACTION  →  token_store.hdf5
# ══════════════════════════════════════════════════════════════════════════════
def extract_token_store(data_dir: str, checkpoint: str, out_dir: str,
                         batch_size: int = 32, device: str = "cpu"):
    """
    Run Bio-PM encoder on every window and save the full (L, 64) token matrix.

    Output HDF5 schema:
        tokens       (N, L, 64)   — raw token embeddings Z
        mean_tokens  (N, 64)      — mean-pooled tokens (for downstream use)
        labels       (N,)         — integer activity labels
        subject_ids  (N,)         — subject IDs
        window_ids   (N,)         — global window index
    """
    print("\n" + "═" * 60)
    print("STEP 1 — Token Extraction")
    print("═" * 60)

    (X, pos_info, add_emb, labels, pids,
     X_grav, raw_acc) = load_preprocessed_h5(data_dir)

    N = X.shape[0]
    print(f"  Loaded {N} windows | ME patches {X.shape} | Gravity {X_grav.shape}")

    dataset = MovementElementDataset(
        X=X, X_grav=raw_acc, y=labels, pos_info=pos_info,
        additional_embedding=add_emb, pid=pids,
        name="token_extract", is_label=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    model = load_pretrained_encoder(checkpoint, device=device)

    all_tokens, all_mean, all_labels, all_pids = [], [], [], []
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            my_X, my_Y, my_PID, raw_batch, my_pos, my_add = batch
            bs = my_X.shape[0]

            my_X  = my_X.to(device,  dtype=torch.float)
            my_pos = my_pos.to(device, dtype=torch.float)
            my_add = my_add.to(device, dtype=torch.float)
            mask = torch.zeros(bs, my_X.shape[1], device=device)

            # Full token matrix Z ∈ R^(B, L, 64)
            tokens = model.encoder_acc(my_X, my_pos, mask, my_add)

            all_tokens.append(tokens.cpu().numpy())
            all_mean.append(tokens.mean(dim=1).cpu().numpy())
            all_labels.append(my_Y.numpy() if isinstance(my_Y, torch.Tensor)
                              else np.array(my_Y))
            all_pids.append(my_PID.numpy() if isinstance(my_PID, torch.Tensor)
                            else np.array(my_PID))
            global_idx += bs

    tokens_arr = np.concatenate(all_tokens, axis=0).astype(np.float32)  # (N, L, 64)
    mean_arr   = np.concatenate(all_mean,   axis=0).astype(np.float32)  # (N, 64)
    labels_arr = np.concatenate(all_labels, axis=0).astype(np.float32)
    pids_arr   = np.concatenate(all_pids,   axis=0).astype(np.float32)
    wids_arr   = np.arange(N, dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    store_path = os.path.join(out_dir, "token_store.hdf5")
    with h5py.File(store_path, "w") as f:
        f.create_dataset("tokens",      data=tokens_arr, compression="gzip")
        f.create_dataset("mean_tokens", data=mean_arr,   compression="gzip")
        f.create_dataset("labels",      data=labels_arr)
        f.create_dataset("subject_ids", data=pids_arr)
        f.create_dataset("window_ids",  data=wids_arr)

    print(f"  Saved token store: {store_path}")
    print(f"    tokens:      {tokens_arr.shape}   (N × L × D)")
    print(f"    mean_tokens: {mean_arr.shape}     (N × D)")
    return tokens_arr, mean_arr, labels_arr, pids_arr


# ══════════════════════════════════════════════════════════════════════════════
# 2.  NEAREST-NEIGHBOUR SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
def nn_sanity_check(mean_tokens: np.ndarray, labels: np.ndarray,
                    n_probe: int = 10, k: int = 5):
    """
    For n_probe random windows, find k nearest neighbours by L2 in token space.
    Reports what fraction of neighbours share the same activity class.
    """
    print("\n" + "═" * 60)
    print("STEP 2 — Nearest-Neighbour Sanity Check")
    print("═" * 60)

    N = mean_tokens.shape[0]
    rng = np.random.default_rng(42)
    probe_ids = rng.choice(N, size=min(n_probe, N), replace=False)

    same_class_fracs = []
    for idx in probe_ids:
        query = mean_tokens[idx]
        dists = np.linalg.norm(mean_tokens - query, axis=1)
        dists[idx] = np.inf                     # exclude self
        nn_ids = np.argsort(dists)[:k]
        nn_labels = labels[nn_ids]
        frac_same = (nn_labels == labels[idx]).mean()
        same_class_fracs.append(frac_same)
        act = ACTIVITY_NAMES.get(int(labels[idx]), f"class_{int(labels[idx])}")
        print(f"  Window {idx:4d} [{act:<12}]  "
              f"NN labels: {[ACTIVITY_NAMES.get(int(l), str(int(l))) for l in nn_labels]}  "
              f"→ {frac_same:.0%} same class")

    mean_frac = np.mean(same_class_fracs)
    print(f"\n  Mean same-class fraction across {n_probe} probes: "
          f"{mean_frac:.1%}")
    if mean_frac >= 0.6:
        print("  ✅ Good — tokens cluster by activity class. Bio-PM is useful!")
    else:
        print("  ⚠️  Low same-class clustering — check preprocessing.")
    return mean_frac


# ══════════════════════════════════════════════════════════════════════════════
# 3.  UMAP — activity colours
# ══════════════════════════════════════════════════════════════════════════════
def plot_activity_umap(mean_tokens: np.ndarray, labels: np.ndarray,
                       out_dir: str):
    print("\n" + "═" * 60)
    print("STEP 3 — UMAP: Token Space coloured by Activity")
    print("═" * 60)

    try:
        import umap
    except ImportError:
        print("  umap-learn not installed. Run: pip install umap-learn")
        print("  Skipping UMAP plot.")
        return

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("  Fitting UMAP (n_neighbors=15, min_dist=0.1) ...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(mean_tokens)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_id, name in ACTIVITY_NAMES.items():
        mask = labels == cls_id
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=ACTIVITY_COLORS[cls_id], label=name,
                   s=18, alpha=0.75, linewidths=0)

    ax.set_title("Bio-PM Token Space — UMAP by Activity", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(title="Activity", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(out_dir, "umap_activity.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return emb


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SUBJECT-STYLE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
def compute_subject_embeddings(mean_tokens: np.ndarray, labels: np.ndarray,
                                pids: np.ndarray, out_dir: str) -> dict:
    """
    For each subject, mean-pool their real token sequences → s ∈ R^D.
    Saves subject_embeddings.npy and prints per-subject stats.
    """
    print("\n" + "═" * 60)
    print("STEP 4 — Subject-Style Embeddings")
    print("═" * 60)

    unique_subjects = sorted(np.unique(pids).astype(int).tolist())
    D = mean_tokens.shape[1]
    subj_embs = {}

    print(f"  {'Subject':>10} {'Windows':>10} {'Norm':>10}")
    print("  " + "-" * 35)
    for sid in unique_subjects:
        mask = pids == sid
        emb  = mean_tokens[mask].mean(axis=0)   # s ∈ R^D
        subj_embs[sid] = emb
        print(f"  {sid:>10} {mask.sum():>10} {np.linalg.norm(emb):>10.4f}")

    # Save as structured array: rows = subjects, cols = embedding dims
    subj_ids  = np.array(sorted(subj_embs.keys()), dtype=np.int32)
    subj_mat  = np.stack([subj_embs[s] for s in subj_ids], axis=0)  # (n_subj, D)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "subject_embeddings.npy"),   subj_mat)
    np.save(os.path.join(out_dir, "subject_embedding_ids.npy"), subj_ids)
    print(f"\n  Saved: {out_dir}/subject_embeddings.npy  shape={subj_mat.shape}")
    print(f"  Saved: {out_dir}/subject_embedding_ids.npy")
    return subj_embs, subj_mat, subj_ids


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UMAP — subject colours
# ══════════════════════════════════════════════════════════════════════════════
def plot_subject_umap(subj_mat: np.ndarray, subj_ids: np.ndarray, out_dir: str):
    print("\n" + "═" * 60)
    print("STEP 5 — UMAP: Subject-Style Embedding Space")
    print("═" * 60)

    if subj_mat.shape[0] < 4:
        print("  Too few subjects for UMAP — skipping (need ≥4).")
        return

    try:
        import umap
    except ImportError:
        print("  umap-learn not installed. Skipping.")
        return

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    reducer = umap.UMAP(n_neighbors=min(5, subj_mat.shape[0] - 1),
                         min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(subj_mat)

    colors = cm.tab20(np.linspace(0, 1, len(subj_ids)))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, sid in enumerate(subj_ids):
        ax.scatter(emb[i, 0], emb[i, 1], c=[colors[i]], s=120,
                   label=f"Subj {sid}", zorder=3)
        ax.annotate(str(sid), (emb[i, 0], emb[i, 1]),
                    fontsize=7, ha='center', va='bottom')

    ax.set_title("Subject-Style Embeddings — UMAP", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    if len(subj_ids) <= 20:
        ax.legend(title="Subject", bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=8, ncol=2)
    plt.tight_layout()

    save_path = os.path.join(out_dir, "umap_subjects.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # Assess separation
    if len(subj_ids) >= 6:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import accuracy_score
        loo = LeaveOneOut()
        preds, truths = [], []
        for tr_i, te_i in loo.split(subj_mat):
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(subj_mat[tr_i], subj_ids[tr_i])
            preds.append(knn.predict(subj_mat[te_i])[0])
            truths.append(subj_ids[te_i][0])
        acc = accuracy_score(truths, preds)
        print(f"  1-NN LOO subject-ID accuracy: {acc:.1%}")
        if acc >= 0.5:
            print("  ✅ Subjects form distinct clusters — subject conditioning is meaningful!")
        else:
            print("  ℹ️  Weak subject separation — activity-only conditioning may suffice.")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BASELINE HAR CLASSIFIER  (real-only, no augmentation)
# ══════════════════════════════════════════════════════════════════════════════
def baseline_har_classifier(features_path: str, out_dir: str):
    """
    Train linear probe + small MLP on Bio-PM features, LOSO cross-validation.
    This is the 'real-only' baseline that CAR-IMU must beat.
    """
    print("\n" + "═" * 60)
    print("STEP 6 — Baseline HAR Classifier (real-only, LOSO)")
    print("═" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import f1_score, classification_report

    data = np.load(features_path)
    X    = data['features'].astype(np.float32)   # (N, 1028)
    y    = data['labels'].astype(int)
    pids = data['pids'].astype(int)

    print(f"  Loaded: {X.shape[0]} windows, {X.shape[1]}-d features")
    print(f"  Subjects: {sorted(np.unique(pids).tolist())}")
    print(f"  Classes:  {sorted(np.unique(y).tolist())}")

    # Class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\n  Class distribution (windows):")
    for cls, cnt in zip(unique_classes, counts):
        print(f"    {ACTIVITY_NAMES.get(cls, str(cls)):<15} {cnt:>6} windows")

    logo = LeaveOneGroupOut()
    subjects = np.unique(pids)

    if len(subjects) < 2:
        print("\n  Only 1 subject found — running train/test split (80/20) instead of LOSO")
        print("  (Re-run after processing all subjects for proper LOSO)")
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Linear probe
        lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                 multi_class='multinomial')
        lr.fit(X_tr, y_tr)
        lr_f1 = f1_score(y_te, lr.predict(X_te), average='macro')

        # MLP
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                             random_state=42, early_stopping=True)
        mlp.fit(X_tr, y_tr)
        mlp_f1 = f1_score(y_te, mlp.predict(X_te), average='macro')

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │  BASELINE RESULTS (80/20 split)     │")
        print(f"  │  Linear probe Macro-F1: {lr_f1:.3f}        │")
        print(f"  │  MLP        Macro-F1: {mlp_f1:.3f}        │")
        print(f"  └─────────────────────────────────────┘")
        print(f"\n  Per-class report (MLP):")
        print(classification_report(
            y_te, mlp.predict(X_te),
            target_names=[ACTIVITY_NAMES.get(i, str(i)) for i in range(6)],
            zero_division=0))

        results = {"linear_f1": float(lr_f1), "mlp_f1": float(mlp_f1),
                   "mode": "train_test_split"}
    else:
        # Full LOSO
        lr_f1s, mlp_f1s = [], []
        for fold, (tr_i, te_i) in enumerate(logo.split(X, y, groups=pids)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_i])
            X_te = scaler.transform(X[te_i])

            lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                     multi_class='multinomial')
            lr.fit(X_tr, y[tr_i])
            lr_f1s.append(f1_score(y[te_i], lr.predict(X_te),
                                    average='macro', zero_division=0))

            mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                  random_state=42, early_stopping=True)
            mlp.fit(X_tr, y[tr_i])
            mlp_f1s.append(f1_score(y[te_i], mlp.predict(X_te),
                                      average='macro', zero_division=0))

            test_subj = np.unique(pids[te_i])[0]
            print(f"  Fold {fold+1:2d} (test subject {test_subj:3d}): "
                  f"LR F1={lr_f1s[-1]:.3f}  MLP F1={mlp_f1s[-1]:.3f}")

        lr_mean,  lr_std  = np.mean(lr_f1s),  np.std(lr_f1s)
        mlp_mean, mlp_std = np.mean(mlp_f1s), np.std(mlp_f1s)

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  BASELINE RESULTS (LOSO, {len(subjects)} subjects)         │")
        print(f"  │  Linear probe  Macro-F1: {lr_mean:.3f} ± {lr_std:.3f}    │")
        print(f"  │  MLP           Macro-F1: {mlp_mean:.3f} ± {mlp_std:.3f}    │")
        print(f"  └─────────────────────────────────────────────┘")
        print(f"\n  ⭐ RECORD THESE — CAR-IMU augmentation must beat {mlp_mean:.3f}")

        results = {
            "linear_f1_mean": float(lr_mean),   "linear_f1_std": float(lr_std),
            "mlp_f1_mean":    float(mlp_mean),   "mlp_f1_std":   float(mlp_std),
            "linear_f1_per_fold": lr_f1s,         "mlp_f1_per_fold": mlp_f1s,
            "mode": "loso",
        }

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "baseline_results.npy"), results)
    print(f"\n  Saved: {out_dir}/baseline_results.npy")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Week 1 analysis — CAR-IMU project")
    p.add_argument("--data_dir",   type=str, default="preprocessed_biopm",
                   help="Directory with Data_MeLabel_*.h5 files")
    p.add_argument("--checkpoint", type=str,
                   default="CS690TR/checkpoints/checkpoint.pt")
    p.add_argument("--features",   type=str,
                   default="features/biopm_features_all.npz",
                   help="Path to .npz from extract_features.py")
    p.add_argument("--out_dir",    type=str, default="results_week1")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--skip_tokens", action="store_true",
                   help="Skip token extraction (use existing token_store.hdf5)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("CAR-IMU  Week 1 Analysis")
    print("=" * 60)
    print(f"  data_dir:   {args.data_dir}")
    print(f"  features:   {args.features}")
    print(f"  out_dir:    {args.out_dir}")

    # ── Step 1: Token extraction ──────────────────────────────────────────────
    store_path = os.path.join(args.out_dir, "token_store.hdf5")
    if args.skip_tokens and os.path.exists(store_path):
        print(f"\nLoading existing token store from {store_path} ...")
        with h5py.File(store_path, "r") as f:
            tokens_arr = f["tokens"][:]
            mean_arr   = f["mean_tokens"][:]
            labels_arr = f["labels"][:]
            pids_arr   = f["subject_ids"][:]
    else:
        tokens_arr, mean_arr, labels_arr, pids_arr = extract_token_store(
            args.data_dir, args.checkpoint, args.out_dir,
            batch_size=args.batch_size, device=args.device)

    # ── Step 2: NN sanity check ───────────────────────────────────────────────
    nn_sanity_check(mean_arr, labels_arr, n_probe=10, k=5)

    # ── Step 3: Activity UMAP ─────────────────────────────────────────────────
    plot_activity_umap(mean_arr, labels_arr, args.out_dir)

    # ── Step 4: Subject embeddings ────────────────────────────────────────────
    subj_embs, subj_mat, subj_ids = compute_subject_embeddings(
        mean_arr, labels_arr, pids_arr, args.out_dir)

    # ── Step 5: Subject UMAP ─────────────────────────────────────────────────
    plot_subject_umap(subj_mat, subj_ids, args.out_dir)

    # ── Step 6: Baseline classifier ───────────────────────────────────────────
    if os.path.exists(args.features):
        baseline_har_classifier(args.features, args.out_dir)
    else:
        print(f"\n  [SKIP] Features file not found: {args.features}")
        print(f"  Run extract_features.py first, then re-run with --skip_tokens")

    print("\n" + "═" * 60)
    print("✅  Week 1 analysis complete!")
    print(f"   All outputs in: {args.out_dir}/")
    print("   token_store.hdf5         ← full Z matrices for CAR-IMU training")
    print("   subject_embeddings.npy   ← subject conditioning vectors")
    print("   umap_activity.png        ← sanity check plot")
    print("   umap_subjects.png        ← subject separation plot")
    print("   baseline_results.npy     ← real-only F1 to beat")
    print("═" * 60)


if __name__ == "__main__":
    main()
