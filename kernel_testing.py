import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import PredefinedSplit, cross_val_score

with open("selected_features.txt", "r") as f:
    # strip whitespace and ignore any empty lines
    selected_cols = [line.strip() for line in f if line.strip()]
    

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
x_train = train_df.drop(columns=['label'])
X_train = x_train[selected_cols].to_numpy()
y_train = train_df['label'].to_numpy()

x_test = test_df.drop(columns=['label'])
X_test = x_test[selected_cols].to_numpy()
y_test = test_df['label'].to_numpy()

# ---------------------------
# 3. 5 fold Cross Validation
# ---------------------------

n_classes = len(data_file_names)
n_folds    = 5
block_size = train_len // n_folds  # = 180

# test_fold[i] = fold-index if sample i is in that fold’s test set
# test_fold = np.empty(len(y_train), dtype=int)

test_fold = np.empty(len(y_train), dtype=int)
for class_id in range(n_classes):
    class_start = class_id * train_len
    for fold in range(n_folds):
        start = class_start + fold * block_size
        end   = class_start + train_len 

        test_fold[start:end] = fold
        
cv = PredefinedSplit(test_fold)

# Define your two kernels to compare
kernels = {
    # "RBF (γ='scale')": SVC(kernel="rbf",    gamma="scale", class_weight="balanced"),
    "Linear": SVC(kernel="linear", class_weight="balanced"),
    # "Poly-2   (degree=2)": SVC(kernel="poly", degree=2,    gamma="scale", class_weight="balanced"),
    # "Poly-3   (degree=3)": SVC(kernel="poly", degree=3,    gamma="scale", class_weight="balanced")
}



for name, clf in kernels.items():
    # build pipeline with scaling
    pipe = make_pipeline(StandardScaler(), clf)

    # get cross-validated predictions on the train set
    y_pred = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=-1)

    # overall macro-F1
    macro_f1 = f1_score(y_train, y_pred, average="macro")
    print(f"\n=== {name} ===")
    print(f"Macro-F1: {macro_f1:.3f}\n")

    # per-class precision/recall/F1
    print(classification_report(
        y_train, y_pred,
        target_names=[str(c) for c in sorted(data_file_names.keys())],
        digits=3
    ))

from sklearn.metrics import classification_report, f1_score, confusion_matrix

# 1) build & fit on the full training set
best_clf = SVC(kernel="linear", class_weight="balanced")
pipe     = make_pipeline(StandardScaler(), best_clf)
pipe.fit(X_train, y_train)

# 2) predict on your test set
y_test_pred = pipe.predict(X_test)

# 3) evaluate
macro_f1_test = f1_score(y_test, y_test_pred, average="macro")
print(f"=== Test set evaluation ===")
print(f"Macro-F1 on test set: {macro_f1_test:.3f}\n")

print(classification_report(
    y_test, y_test_pred,
    target_names=[str(c) for c in sorted(data_file_names.keys())],
    digits=3
))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 1) compute the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# 2) set up class labels (as strings)
classes = [str(c) for c in sorted(data_file_names.keys())]

# 3) plot
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')           # default colormap (viridis)
ax.figure.colorbar(im, ax=ax)

# 4) ticks and labels
ax.set(
    xticks=range(len(classes)), 
    yticks=range(len(classes)),
    xticklabels=classes, 
    yticklabels=classes,
    xlabel='Predicted label', 
    ylabel='True label',
    title='Confusion Matrix Heatmap'
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 5) annotation
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, 
            format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

fig.tight_layout()
plt.show()

