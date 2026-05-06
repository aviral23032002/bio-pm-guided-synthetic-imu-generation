import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from car_imu_decoder import ACTIVITY_NAMES, NUM_CLASSES

LABELS = [ACTIVITY_NAMES[i] for i in range(NUM_CLASSES)]

# Load tokens
with h5py.File("results_wisdm_v2_6class/token_store.hdf5", "r") as f:
    tokens  = f["tokens"][:].astype(np.float32)
    labels  = f["labels"][:].astype(int)
    pids    = f["subject_ids"][:].astype(int)

# 64-d global mean
feats_64 = tokens.mean(axis=1)

# Simple subject-based split — last 5 subjects as test
test_subjects = np.unique(pids)[-5:]
test_mask  = np.isin(pids, test_subjects)
train_mask = ~test_mask

sc = StandardScaler().fit(feats_64[train_mask])
X_train = sc.transform(feats_64[train_mask])
X_test  = sc.transform(feats_64[test_mask])
y_train = labels[train_mask]
y_test  = labels[test_mask]

# Train LR
lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                        random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, preds, normalize="true")

# Plot
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=LABELS, yticklabels=LABELS,
    vmin=0, vmax=1, ax=ax, annot_kws={"size": 10})
ax.set_title("Confusion Matrix (Normalized) — 64-d Global Mean\n"
             "Showing representation gap", fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted label", fontsize=10)
ax.set_ylabel("True label",      fontsize=10)
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig("cm_64d.png", dpi=150, bbox_inches="tight")
print("Saved: cm_64d.png")