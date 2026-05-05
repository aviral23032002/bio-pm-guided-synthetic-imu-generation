import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold

# ── Setup ─────────────────────────────────────────────────────────────────────
REAL_TOKENS = "results_wisdm_v2_6class/token_store.hdf5"
OUT_DIR = "analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

ACTIVITY_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]

def sanitize_features(feats: np.ndarray, name: str = "") -> np.ndarray:
    """Replace NaN/Inf with 0 and clip extreme values."""
    n_bad = (~np.isfinite(feats)).sum()
    if n_bad > 0:
        print(f"  ⚠  {name}: {n_bad} non-finite values replaced with 0")
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.clip(feats, -30.0, 30.0)
    return feats.astype(np.float32)

def compute_320d_features(tokens, gravity):
    """Axis-wise pooling + global std + gravity."""
    m_x = tokens[:, 0::3, :].mean(axis=1)
    m_y = tokens[:, 1::3, :].mean(axis=1)
    m_z = tokens[:, 2::3, :].mean(axis=1)
    std = tokens.std(axis=1)
    feats = np.concatenate([m_x, m_y, m_z, std, gravity], axis=1)
    return sanitize_features(feats, "320-d features")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print(f"Loading real tokens from {REAL_TOKENS}...")
with h5py.File(REAL_TOKENS, 'r') as f:
    tokens  = f['tokens'][:].astype(np.float32)
    labels  = f['labels'][:].astype(int)
    pids    = f['subject_ids'][:].astype(int)
    gravity = f['gravity_vecs'][:].astype(np.float32)

print(f"Computing 320-d features...")
features = compute_320d_features(tokens, gravity)
print(f"Features shape: {features.shape}")

# ── 2. Confusion Matrix (LOSO approx via GroupKFold) ─────────────────────────
print("\nRunning Cross-Validation for Confusion Matrix...")
gkf = GroupKFold(n_splits=5)
all_preds = []
all_true = []

for train_idx, test_idx in gkf.split(features, labels, groups=pids):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)
    
    print(f"  Fold {len(all_preds)//2500 + 1}: X_train_sc mean={X_train_sc.mean():.2f}, std={X_train_sc.std():.2f}")
    
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial", random_state=42)
    clf.fit(X_train_sc, y_train)
    
    preds = clf.predict(X_test_sc)
    all_preds.extend(preds)
    all_true.extend(y_test)

cm = confusion_matrix(all_true, all_preds, normalize='true')
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIVITY_NAMES)
disp.plot(cmap='Blues', ax=ax, values_format='.2f')
plt.title("Confusion Matrix (Normalized) - 320-d Features", fontsize=14)
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
print(f"  Confusion Matrix saved: {os.path.join(OUT_DIR, 'confusion_matrix.png')}")

# ── 3. Visualization (t-SNE) ──────────────────────────────────────────────────
print("\nRunning t-SNE visualization (subset of 3000 windows)...")
# Subset for speed
np.random.seed(42)
subset_idx = np.random.choice(len(features), 3000, replace=False)
X_subset = features[subset_idx]
y_subset = labels[subset_idx]

sc = StandardScaler()
X_subset_sc = sc.fit_transform(X_subset)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb = tsne.fit_transform(X_subset_sc)

plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', 6)
for i, name in enumerate(ACTIVITY_NAMES):
    mask = y_subset == i
    plt.scatter(emb[mask, 0], emb[mask, 1], label=name, alpha=0.6, s=15, color=colors(i))

plt.title("t-SNE Overlay: 320-d Features (Bio-PM Tokens)", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(OUT_DIR, "tsne_overlay.png"), dpi=200)
print(f"  t-SNE Plot saved: {os.path.join(OUT_DIR, 'tsne_overlay.png')}")

# ── 4. Analyze Walking Overlap ───────────────────────────────────────────────
walking_idx = 0
confused_with = cm[walking_idx].argsort()[::-1]
print(f"\nWalking Confusion Analysis:")
for idx in confused_with:
    if idx == walking_idx: continue
    print(f"  Confused with {ACTIVITY_NAMES[idx]:<12}: {cm[walking_idx, idx]:.2%}")

print("\nAnalysis Complete.")
