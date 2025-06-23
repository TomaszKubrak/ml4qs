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

pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate
print("\n--- Test Set Evaluation ---")
print(f"Test accuracy: {pipeline.score(X_test, y_test):.3f}\n")

print("Classification report (per class):")
print(classification_report(y_test, y_pred, target_names=list(data_file_names.values())))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import f1_score

f1_macro_test = f1_score(y_test, y_pred, average="macro")
print(f"Test F1 macro: {f1_macro_test:.3f}")