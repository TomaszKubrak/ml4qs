import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit, cross_val_score



data_file_names = {0:"cycling_features.csv", 1:"gym_features.csv", 2:"home_features.csv",
                   3:"park_features.csv", 4:"public_transport_features.csv",
                   5:"shop_features.csv", 6:"street_features.csv", 7:"uni_features.csv"}

all_train = []
all_test = []

train_len = 900   # in seconds (1 window = 1s or 6s depending on your setup)
test_len = 260

for label, file_name in data_file_names.items():
    df = pd.read_csv("./dataset/" + file_name).iloc[:1160].copy()
    df["label"] = label

    # split into train/test segments (first 900s, last 260s)
    train_df = df.iloc[:train_len].copy()
    test_df  = df.iloc[train_len:].copy()

    all_train.append(train_df)
    all_test.append(test_df)

# Combine all
train_df = pd.concat(all_train, ignore_index=True)
test_df  = pd.concat(all_test, ignore_index=True)

# ----------------------------
# 2. Features and labels
# ----------------------------
X_train = train_df.drop(columns=['label']).to_numpy()
y_train = train_df['label'].to_numpy()

X_test = test_df.drop(columns=['label']).to_numpy()
y_test = test_df['label'].to_numpy()

# Feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# 1. Scale your data
scaler = StandardScaler().fit(X_train)
Xs = scaler.transform(X_train)

# 2. Fit a LinearSVC with L1 penalty (forces sparsity)
lsvc = LinearSVC(
    penalty="l1",
    dual=False,
    C=0.0009,                      # you can grid-search this
    class_weight="balanced",
    max_iter=10_000,
    random_state=42
).fit(Xs, y_train)

# 3. Select features whose coefficients are nonzero
sfm = SelectFromModel(lsvc, prefit=True)
mask = sfm.get_support()       # boolean mask of selected features

# 4. Map back to your original feature names
feature_names = train_df.drop(columns="label").columns
selected = feature_names[mask]

print(f"Kept {len(selected)} / {len(feature_names)} features:")
print(len(selected.tolist()))

lines = []
lines.extend(selected.tolist())  # one feature name per line

# Write them out:
with open("selected_features.txt", "w") as f:
    f.write("\n".join(lines))



# ---------------------------
# 3. 5 fold Cross Validation
# ---------------------------

# n_classes = len(data_file_names)
# n_folds    = 5
# block_size = train_len // n_folds  # = 180

# # test_fold[i] = fold-index if sample i is in that fold’s test set
# # test_fold = np.empty(len(y_train), dtype=int)

# test_fold = np.empty(len(y_train), dtype=int)
# for class_id in range(n_classes):
#     class_start = class_id * train_len
#     for fold in range(n_folds):
#         start = class_start + fold * block_size
#         end   = class_start + train_len 

#         test_fold[start:end] = fold
        
# cv = PredefinedSplit(test_fold)

# # 3. Define your SVM pipeline
# pipeline = make_pipeline(
#     StandardScaler(),
#     SVC(kernel="rbf", class_weight="balanced")
# )

# # 4. Run cross_val_score on the training set
# scores = cross_val_score(
#     pipeline, X_train, y_train,
#     cv=cv,
#     scoring="f1_macro",
#     n_jobs=-1
# )

# print("5-fold CV f1_macro scores:", np.round(scores, 3))
# print("Mean f1_macro:", np.round(scores.mean(), 3))


#feature selection to prevent overfitting - forward vs backward & regularization?


