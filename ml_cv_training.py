import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 0. Config
# ----------------------------
data_file_names = {
    0: "cycling_features.csv",
    1: "gym_features.csv",
    2: "home_features.csv",
    3: "park_features.csv",
    4: "public_transport_features.csv",
    5: "shop_features.csv",
    6: "street_features.csv",
    7: "uni_features.csv"
}

train_len = 900 + 260  # full training duration
test_len  = 300        # first 5 minutes

# ----------------------------
# 1. Load train/test data
# ----------------------------
all_train = []
all_test  = []

for label, fn in data_file_names.items():
    df_train = pd.read_csv(f"./dataset_train/{fn}").iloc[:train_len].copy()
    df_train["label"] = label

    df_test = pd.read_csv(f"./dataset_test/{fn}").iloc[:test_len].copy()
    df_test["label"] = label

    all_train.append(df_train)
    all_test.append(df_test)

train_df = pd.concat(all_train, ignore_index=True)
test_df  = pd.concat(all_test,  ignore_index=True)

# ----------------------------
# 2. Load selected features
# ----------------------------
with open("selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"Using {len(selected_features)} selected features")

# ----------------------------
# 3. Extract selected features + labels
# ----------------------------
X_train = train_df[selected_features].to_numpy()
y_train = train_df['label'].to_numpy()

X_test  = test_df[selected_features].to_numpy()
y_test  = test_df['label'].to_numpy()

# ----------------------------
# 4. PredefinedSplit: 5-fold CV
# ----------------------------
n_classes  = len(data_file_names)
n_folds    = 5
block_size = train_len // n_folds

test_fold = np.empty(len(y_train), dtype=int)
for class_id in range(n_classes):
    base = class_id * train_len
    for fold in range(n_folds):
        start = base + fold * block_size
        end   = base + (fold + 1) * block_size
        test_fold[start:end] = fold

cv = PredefinedSplit(test_fold)

# ----------------------------
# 5. Model + cross-validation
# ----------------------------
# pipeline = make_pipeline(
#     StandardScaler(),
#     SVC(kernel="rbf", class_weight="balanced"),
# )
# pipeline = make_pipeline(
#     StandardScaler(),
#     SVC(kernel="rbf",    gamma="scale", class_weight="balanced"),
# )

pipeline = make_pipeline(
    RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",  # good for imbalanced datasets
        random_state=42,
        n_jobs=-1                  # use all CPU cores
    )
)
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

print("5-fold CV f1_macro scores:", np.round(scores, 3))
print("Mean f1_macro:", np.round(scores.mean(), 3))

# ------------------------------------------------------------------------------
# … everything up through pipeline.fit(X_train, y_train) stays the same …
# ------------------------------------------------------------------------------

pipeline.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 6. Predict in 30 s intervals on the test set
# ------------------------------------------------------------------------------
interval_size = 30                 # rows per 30 s
n_intervals_per_class = test_len // interval_size
n_classes = len(data_file_names)

y_chunk_true = []
y_chunk_pred = []

for class_id in range(n_classes):
    # compute start of this class’s block in the concatenated test set
    base = class_id * test_len
    for i in range(n_intervals_per_class):
        start = base + i * interval_size
        end   = start + interval_size

        # true labels in this chunk (all the same, since df_test["label"] was constant)
        true_labels = y_test[start:end]
        y_chunk_true.append(true_labels[0])

        # per‐sample predictions, then majority vote
        preds = pipeline.predict(X_test[start:end])
        # majority vote: take the label with highest count
        majority_label = np.bincount(preds).argmax()
        y_chunk_pred.append(majority_label)

# ------------------------------------------------------------------------------
# 7. Evaluate chunk-level predictions
# ------------------------------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix, f1_score

print("\n--- 30 s-chunk Evaluation (80 total chunks) ---")
print(f"Overall chunk-accuracy: {np.mean(np.array(y_chunk_pred) == np.array(y_chunk_true)):.3f}\n")

# per-class reports (names are the same as before)
print("Classification report (per class):")
print(classification_report(y_chunk_true, y_chunk_pred,
                            target_names=list(data_file_names.values())))

print("Confusion matrix:")
print(confusion_matrix(y_chunk_true, y_chunk_pred))

f1_macro_chunks = f1_score(y_chunk_true, y_chunk_pred, average="macro")
print(f"Chunk F1 macro: {f1_macro_chunks:.3f}")

import seaborn as sns
import matplotlib.pyplot as plt

# Create the confusion matrix
cm = confusion_matrix(y_chunk_true, y_chunk_pred)

# Define class names
class_names = [name.replace("_features.csv", "") for name in data_file_names.values()]

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (30s Chunk-Level Predictions)")
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.jpg", dpi=300)
plt.show()
