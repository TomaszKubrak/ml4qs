import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit, cross_val_score

# mapping of label → filename
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

# how many seconds (rows) to keep
train_len = 900+260   # first 900 s of each train recording
test_len  = 300   # first 300 s (5 min) of each test recording

all_train = []
all_test  = []

for label, fn in data_file_names.items():
    # --- load & crop train ---
    df_train = (
        pd.read_csv(f"./dataset_train/{fn}")
          .iloc[:train_len]
          .copy()
    )
    df_train["label"] = label

    # --- load & crop test ---
    df_test = (
        pd.read_csv(f"./dataset_test/{fn}")
          .iloc[:test_len]
          .copy()
    )
    df_test["label"] = label

    all_train.append(df_train)
    all_test.append(df_test)

# combine into single DataFrames
train_df = pd.concat(all_train, ignore_index=True)
test_df  = pd.concat(all_test,  ignore_index=True)

# ----------------------------
# 2. Features and labels
# ----------------------------
X_train = train_df.drop(columns=['label']).to_numpy()
y_train = train_df['label'].to_numpy()

X_test  = test_df.drop(columns=['label']).to_numpy()
y_test  = test_df['label'].to_numpy()

# ----------------------------
# 3. Feature selection via L1‐LinearSVC
# ----------------------------
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

# 1. scale
scaler = StandardScaler().fit(X_train)
Xs = scaler.transform(X_train)

# 2. L1‐penalized SVC
lsvc = LinearSVC(
    penalty="l1",
    dual=False,
    C=0.0009,
    class_weight="balanced",
    max_iter=10_000,
    random_state=42
).fit(Xs, y_train)

# 3. select non‐zero coeff features
sfm  = SelectFromModel(lsvc, prefit=True)
mask = sfm.get_support()

feature_names = train_df.drop(columns="label").columns
selected      = feature_names[mask]

print(f"Kept {len(selected)} / {len(feature_names)} features:")
print(selected.tolist())

# write out
with open("selected_features.txt", "w") as f:
    f.write("\n".join(selected.tolist()))