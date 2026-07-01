"""
Assignment 3 - Face Recognition (PCA, LDA, KNN + Bonus)

This file covers the main assignment and the bonus parts:
1) Load ORL/AT&T face dataset
2) Build data matrix D and labels y
3) 50% split: odd rows for training, even rows for testing
4) PCA for alpha = 0.80, 0.85, 0.90, 0.95
5) Multiclass LDA using 39 dominant eigenvectors
6) KNN tuning for k = 1, 3, 5, 7
7) Bonus A: face vs non-face classification
8) Bonus B: 7 train / 3 test split comparison

Before running:
- Change BASE_PATH to the folder that contains s1, s2, ..., s40
- For the bonus, put non-face images in NON_FACE_PATH
  Non-face images can be jpg, jpeg, png, bmp, pgm. The code resizes them to 92x112.
"""

import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# Settings
# =========================

BASE_PATH = r"E:\semester 4\machine learing\assigments\ass3"
NON_FACE_PATH = r"E:\semester 4\machine learing\assigments\ass3\non_faces"
OUTPUT_DIR = "output_plots"

IMAGE_SIZE = (92, 112)        # width, height
N_FEATURES = 92 * 112
ALPHAS = [0.80, 0.85, 0.90, 0.95]
K_VALUES = [1, 3, 5, 7]


# =========================
# Loading data
# =========================

def find_orl_folder(base_path):
    """Try common folders until finding s1 and s40."""
    possible_paths = [
        base_path,
        os.path.join(base_path, "att_faces"),
        os.path.join(base_path, "archive"),
        os.path.join(base_path, "archive", "att_faces"),
    ]

    for p in possible_paths:
        if os.path.isdir(os.path.join(p, "s1")) and os.path.isdir(os.path.join(p, "s40")):
            return p

    print("Could not find the ORL dataset. I checked:")
    for p in possible_paths:
        print(" -", p)
    return None


def load_orl_dataset(dataset_path):
    """Load the 40 subjects, 10 images each, and return D and y."""
    D_rows = []
    y_rows = []

    for subject_id in range(1, 41):
        subject_dir = os.path.join(dataset_path, f"s{subject_id}")
        if not os.path.isdir(subject_dir):
            raise FileNotFoundError(f"Missing folder: {subject_dir}")

        for img_id in range(1, 11):
            img_path = os.path.join(subject_dir, f"{img_id}.pgm")
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")

            img = Image.open(img_path).convert("L")
            img = img.resize(IMAGE_SIZE)
            vec = np.array(img, dtype=np.float64).flatten()

            D_rows.append(vec)
            y_rows.append(subject_id)

    D = np.array(D_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=int)

    print("Loaded ORL dataset")
    print("D shape:", D.shape)
    print("y shape:", y.shape)
    return D, y


def load_non_face_images(folder_path):
    """Load non-face images recursively and resize them to 92x112."""
    if not os.path.isdir(folder_path):
        print("\nNon-face folder was not found:", folder_path)
        print("Bonus A will be skipped. Put non-face images in this folder to run it.")
        return np.empty((0, N_FEATURES), dtype=np.float64)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".pgm")
    vectors = []

    for root, _, files in os.walk(folder_path):
        for name in files:
            if name.lower().endswith(valid_ext):
                path = os.path.join(root, name)
                try:
                    img = Image.open(path).convert("L")
                    img = img.resize(IMAGE_SIZE)
                    vectors.append(np.array(img, dtype=np.float64).flatten())
                except Exception as e:
                    print("Skipped image:", path, "because", e)

    X = np.array(vectors, dtype=np.float64)
    print("Loaded non-face images:", X.shape[0])
    return X


# =========================
# Splits
# =========================

def split_odd_even_rows(D, y):
    """
    Assignment split:
    row 1,3,5,... train
    row 2,4,6,... test
    In Python this means index 0,2,4,... train.
    """
    train_mask = np.arange(len(y)) % 2 == 0
    return D[train_mask], y[train_mask], D[~train_mask], y[~train_mask]


def split_7_train_3_test(D, y):
    """
    Bonus B split:
    For each subject: first 7 images for training and last 3 images for testing.
    The ORL data is stored as s1 images 1..10, then s2 images 1..10, etc.
    """
    train_indices = []
    test_indices = []

    for subject in range(40):
        start = subject * 10
        train_indices.extend(range(start, start + 7))
        test_indices.extend(range(start + 7, start + 10))

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return D[train_indices], y[train_indices], D[test_indices], y[test_indices]


# =========================
# PCA and LDA
# =========================

def compute_pca_projection(X_train, alpha):
    """Return PCA projection matrix U, mean vector, and number of components."""
    mu = X_train.mean(axis=0)
    Z = X_train - mu

    # SVD is used instead of making the huge covariance matrix directly.
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)

    eigenvalues = S ** 2
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    r = int(np.searchsorted(cum_var, alpha) + 1)
    r = min(r, len(eigenvalues))

    U = Vt[:r].T
    return U, mu, r, cum_var


def compute_lda_projection(X_train, y_train, n_components=None):
    """
    Multiclass LDA.

    The maximum number of useful LDA dimensions is number_of_classes - 1.
    For the 40 subject face recognition task, this gives 39 dimensions.
    For face vs non-face bonus, this gives only 1 dimension.

    I used PCA before LDA because the image feature size is very large and Sw
    becomes singular if we apply LDA directly.
    """
    classes = np.unique(y_train)
    n_samples = X_train.shape[0]
    n_classes = len(classes)

    max_lda_dims = n_classes - 1
    if n_components is None:
        n_components = max_lda_dims
    lda_dims = min(n_components, max_lda_dims)

    # PCA step before LDA (Fisherfaces idea)
    pca_dims = min(n_samples - n_classes, n_samples - 1)
    if pca_dims < 1:
        raise ValueError("Not enough training samples for LDA.")

    mu = X_train.mean(axis=0)
    Z = X_train - mu
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Vt[:pca_dims].T
    X_pca = Z @ P

    overall_mean = X_pca.mean(axis=0)
    Sw = np.zeros((pca_dims, pca_dims), dtype=np.float64)
    Sb = np.zeros((pca_dims, pca_dims), dtype=np.float64)

    for c in classes:
        X_c = X_pca[y_train == c]
        mean_c = X_c.mean(axis=0)

        Z_c = X_c - mean_c
        Sw += Z_c.T @ Z_c

        diff = (mean_c - overall_mean).reshape(-1, 1)
        Sb += X_c.shape[0] * (diff @ diff.T)

    # small value to avoid numerical problems
    Sw = Sw + 1e-6 * np.eye(Sw.shape[0])

    M = np.linalg.pinv(Sw) @ Sb
    eigvals, eigvecs = np.linalg.eig(M)

    eigvals = eigvals.real
    eigvecs = eigvecs.real

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    W_pca = eigvecs[:, :lda_dims]
    W = P @ W_pca
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

    return W, mu, lda_dims


def project(X, U, mu):
    return (X - mu) @ U


# =========================
# KNN
# =========================

def knn_predict(X_train, y_train, X_test, k=1):
    """Simple KNN using Euclidean distance."""
    sq_train = np.sum(X_train ** 2, axis=1)
    sq_test = np.sum(X_test ** 2, axis=1)
    dists = sq_test[:, None] + sq_train[None, :] - 2 * (X_test @ X_train.T)
    dists = np.maximum(dists, 0)

    neighbors = np.argsort(dists, axis=1)[:, :k]
    preds = []

    for row in neighbors:
        labels = y_train[row]
        counts = Counter(labels)
        max_votes = max(counts.values())
        tied = {lab for lab, cnt in counts.items() if cnt == max_votes}

        # tie break: choose the nearest label among tied labels
        chosen = None
        for lab in labels:
            if lab in tied:
                chosen = lab
                break
        preds.append(chosen)

    return np.array(preds)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred) * 100)


# =========================
# Tables and plots
# =========================

def write_csv(rows, path, columns):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")


def print_table(rows, title):
    print("\n" + title)
    print("-" * 85)
    for row in rows:
        print(row)
    print("-" * 85)


def plot_line(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pca_alpha(rows, save_path):
    selected = [r for r in rows if r["Method"] == "PCA" and r["K"] == 1]
    x = [str(r["Alpha"]) for r in selected]
    y = [r["Accuracy"] for r in selected]
    plot_line(x, y, "PCA Accuracy vs Alpha (K=1)", "Alpha", "Accuracy (%)", save_path)


def plot_knn_for_method(rows, method, save_path, alpha=None):
    selected = [r for r in rows if r["Method"] == method]
    if alpha is not None:
        selected = [r for r in selected if r["Alpha"] == alpha]

    x = [r["K"] for r in selected]
    y = [r["Accuracy"] for r in selected]
    plot_line(x, y, f"{method} Accuracy vs K", "K", "Accuracy (%)", save_path)


def save_success_failure_examples(X_test, y_test, y_pred, title, save_path, max_cases=5):
    """Save a small figure containing correct and wrong predictions."""
    correct_idx = np.where(y_test == y_pred)[0][:max_cases]
    wrong_idx = np.where(y_test != y_pred)[0][:max_cases]

    rows = []
    for idx in correct_idx:
        rows.append((idx, "Success"))
    for idx in wrong_idx:
        rows.append((idx, "Failure"))

    if len(rows) == 0:
        print("No cases to plot for", title)
        return

    plt.figure(figsize=(2.2 * len(rows), 3))
    for i, (idx, status) in enumerate(rows):
        img = X_test[idx].reshape(112, 92)
        plt.subplot(1, len(rows), i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{status}\nT:{y_test[idx]} P:{y_pred[idx]}", fontsize=8)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()



def save_mean_and_eigenfaces(X_train, save_path, alpha=0.95, n_show=8):
    """Save mean face and first eigenfaces from PCA."""
    U, mu, r, _ = compute_pca_projection(X_train, alpha)
    n_show = min(n_show, U.shape[1])

    plt.figure(figsize=(2.2 * (n_show + 1), 3))

    plt.subplot(1, n_show + 1, 1)
    plt.imshow(mu.reshape(112, 92), cmap="gray")
    plt.title("Mean")
    plt.axis("off")

    for i in range(n_show):
        ef = U[:, i]
        ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-12)
        plt.subplot(1, n_show + 1, i + 2)
        plt.imshow(ef.reshape(112, 92), cmap="gray")
        plt.title("EF " + str(i + 1))
        plt.axis("off")

    plt.suptitle("Mean Face and Eigenfaces")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)


def save_sample_faces(D, y, save_path, n_subjects=10):
    """Save a quick sample grid from the dataset."""
    plt.figure(figsize=(10, 3))
    for i in range(n_subjects):
        idx = i * 10
        plt.subplot(1, n_subjects, i + 1)
        plt.imshow(D[idx].reshape(112, 92), cmap="gray")
        plt.title("s" + str(y[idx]))
        plt.axis("off")
    plt.suptitle("Sample ORL Face Images")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)


def save_confusion_matrix(y_true, y_pred, title, save_path, n_classes=40):
    """Save confusion matrix without using sklearn."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 1 <= int(t) <= n_classes and 1 <= int(p) <= n_classes:
            cm[int(t) - 1, int(p) - 1] += 1

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted subject")
    plt.ylabel("True subject")
    plt.colorbar()
    ticks = np.arange(n_classes)
    plt.xticks(ticks, [str(i + 1) for i in ticks], fontsize=6, rotation=90)
    plt.yticks(ticks, [str(i + 1) for i in ticks], fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)


def get_best_pca_predictions(X_train, y_train, X_test, y_test):
    """Return predictions for the best PCA setting among alpha and K values."""
    best = None
    for alpha in ALPHAS:
        U, mu, r, _ = compute_pca_projection(X_train, alpha)
        Xtr = project(X_train, U, mu)
        Xte = project(X_test, U, mu)
        for k in K_VALUES:
            pred = knn_predict(Xtr, y_train, Xte, k)
            acc = accuracy(y_test, pred)
            if best is None or acc > best["acc"]:
                best = {"alpha": alpha, "k": k, "r": r, "pred": pred, "acc": acc}
    return best


def get_lda_predictions(X_train, y_train, X_test, y_test, k=1):
    """Return LDA predictions for one K value."""
    W, mu, dims = compute_lda_projection(X_train, y_train, n_components=39)
    Xtr = project(X_train, W, mu)
    Xte = project(X_test, W, mu)
    pred = knn_predict(Xtr, y_train, Xte, k)
    acc = accuracy(y_test, pred)
    return {"k": k, "dims": dims, "pred": pred, "acc": acc}


def plot_split_comparison(rows_50, rows_73, save_path):
    """Compare best PCA and LDA results for the two splits."""
    best_50_pca = max([r for r in rows_50 if r["Method"] == "PCA"], key=lambda r: r["Accuracy"])
    best_50_lda = max([r for r in rows_50 if r["Method"] == "LDA"], key=lambda r: r["Accuracy"])
    best_73_pca = max([r for r in rows_73 if r["Method"] == "PCA"], key=lambda r: r["Accuracy"])
    best_73_lda = max([r for r in rows_73 if r["Method"] == "LDA"], key=lambda r: r["Accuracy"])

    labels = ["PCA 50%", "LDA 50%", "PCA 7/3", "LDA 7/3"]
    values = [best_50_pca["Accuracy"], best_50_lda["Accuracy"], best_73_pca["Accuracy"], best_73_lda["Accuracy"]]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Best Accuracy: 50% Split vs 7/3 Split")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=20)
    for i, v in enumerate(values):
        plt.text(i, v + 1, str(v) + "%", ha="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)

# =========================
# Main experiments
# =========================

def run_pca_lda_experiments(X_train, y_train, X_test, y_test, split_name):
    """Run PCA, LDA, and KNN tuning for one split."""
    rows = []

    print("\nRunning PCA for", split_name)
    for alpha in ALPHAS:
        U, mu, r, _ = compute_pca_projection(X_train, alpha)
        Xtr = project(X_train, U, mu)
        Xte = project(X_test, U, mu)

        for k in K_VALUES:
            pred = knn_predict(Xtr, y_train, Xte, k)
            acc = accuracy(y_test, pred)
            rows.append({
                "Split": split_name,
                "Method": "PCA",
                "Alpha": alpha,
                "K": k,
                "Components": r,
                "Accuracy": round(acc, 2),
            })

    print("Running LDA for", split_name)
    W, mu, dims = compute_lda_projection(X_train, y_train, n_components=39)
    Xtr = project(X_train, W, mu)
    Xte = project(X_test, W, mu)

    for k in K_VALUES:
        pred = knn_predict(Xtr, y_train, Xte, k)
        acc = accuracy(y_test, pred)
        rows.append({
            "Split": split_name,
            "Method": "LDA",
            "Alpha": "-",
            "K": k,
            "Components": dims,
            "Accuracy": round(acc, 2),
        })

    return rows


def run_bonus_non_faces(D_faces, non_faces):
    """
    Bonus A:
    Binary classification between face and non-face images.
    Face images are fixed. Non-face count changes.
    """
    if non_faces.shape[0] < 10:
        print("\nNot enough non-face images for Bonus A. Add more images to NON_FACE_PATH.")
        return []

    print("\nRunning Bonus A: faces vs non-faces")

    # fixed face split using the same odd/even idea
    y_faces_binary = np.ones(D_faces.shape[0], dtype=int)       # face = 1
    X_face_train, y_face_train, X_face_test, y_face_test = split_odd_even_rows(D_faces, y_faces_binary)

    max_each_side = non_faces.shape[0] // 2
    candidate_counts = [20, 40, 80, 120, 160, 200]
    counts = [c for c in candidate_counts if c <= max_each_side]

    if len(counts) == 0:
        counts = [max_each_side]

    rows = []
    best_pred = None
    best_X_test = None
    best_y_test = None
    best_acc = -1

    for n in counts:
        X_nf_train = non_faces[:n]
        X_nf_test = non_faces[n:2*n]
        y_nf_train = np.zeros(X_nf_train.shape[0], dtype=int)   # non-face = 0
        y_nf_test = np.zeros(X_nf_test.shape[0], dtype=int)

        X_train = np.vstack([X_face_train, X_nf_train])
        y_train = np.concatenate([y_face_train, y_nf_train])
        X_test = np.vstack([X_face_test, X_nf_test])
        y_test = np.concatenate([y_face_test, y_nf_test])

        # PCA binary classification
        U, mu, r, _ = compute_pca_projection(X_train, alpha=0.95)
        Xtr = project(X_train, U, mu)
        Xte = project(X_test, U, mu)
        pred_pca = knn_predict(Xtr, y_train, Xte, k=1)
        acc_pca = accuracy(y_test, pred_pca)

        # LDA binary classification: dominant eigenvectors = classes - 1 = 1
        W, mu_lda, dims = compute_lda_projection(X_train, y_train, n_components=1)
        Xtr_lda = project(X_train, W, mu_lda)
        Xte_lda = project(X_test, W, mu_lda)
        pred_lda = knn_predict(Xtr_lda, y_train, Xte_lda, k=1)
        acc_lda = accuracy(y_test, pred_lda)

        rows.append({
            "NonFaceTrainCount": n,
            "FixedFaceTrainCount": len(X_face_train),
            "PCA_Components": r,
            "PCA_Accuracy": round(acc_pca, 2),
            "LDA_Components": dims,
            "LDA_Accuracy": round(acc_lda, 2),
        })

        if acc_lda > best_acc:
            best_acc = acc_lda
            best_pred = pred_lda
            best_X_test = X_test
            best_y_test = y_test

    # Plot accuracy vs number of non-face images
    x = [r["NonFaceTrainCount"] for r in rows]
    y_pca = [r["PCA_Accuracy"] for r in rows]
    y_lda = [r["LDA_Accuracy"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y_pca, marker="o", label="PCA")
    plt.plot(x, y_lda, marker="o", label="LDA")
    plt.title("Faces vs Non-Faces Accuracy")
    plt.xlabel("Number of non-face images in training")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bonus_faces_vs_nonfaces_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.show()

    save_success_failure_examples(
        best_X_test,
        best_y_test,
        best_pred,
        "Bonus A: Success and Failure Cases (0=Non-face, 1=Face)",
        os.path.join(OUTPUT_DIR, "bonus_success_failure_cases.png")
    )

    print("\nFor Bonus A LDA, number of dominant eigenvectors used = 1")
    print("Reason: it is a binary problem, so LDA can use C - 1 = 2 - 1 = 1 dimension.")

    return rows


def write_short_report_notes(main_rows, bonus_rows):
    """Save a small text file with comments that can be used in the report."""
    path = os.path.join(OUTPUT_DIR, "report_notes.txt")

    best_pca = max([r for r in main_rows if r["Method"] == "PCA"], key=lambda r: r["Accuracy"])
    best_lda = max([r for r in main_rows if r["Method"] == "LDA"], key=lambda r: r["Accuracy"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("Report Notes\n")
        f.write("============\n\n")
        f.write("1) Relation between alpha and PCA accuracy:\n")
        f.write("When alpha increases, PCA keeps more variance from the original images. This may improve accuracy because more information is kept. However, after a certain alpha, the improvement can become small because extra components may include noise or less useful details.\n\n")

        f.write("2) PCA vs LDA:\n")
        f.write(f"Best PCA result in the 50% split: {best_pca}\n")
        f.write(f"Best LDA result in the 50% split: {best_lda}\n")
        f.write("LDA is supervised, so it uses class labels and tries to separate classes. PCA is unsupervised and only keeps variance, so high variance does not always mean better class separation.\n\n")

        f.write("3) Bonus A - dominant eigenvectors for LDA:\n")
        f.write("For faces vs non-faces, the number of classes is 2. LDA uses C - 1 dominant eigenvectors, so the number is 1.\n\n")

        f.write("4) Criticism of accuracy for many non-face images:\n")
        f.write("Accuracy can be misleading if the dataset becomes imbalanced. For example, if there are too many non-face images compared to face images, a classifier may get high accuracy by predicting the majority class more often. It is better to also check confusion matrix, precision, recall, and examples of success/failure cases.\n\n")

        if bonus_rows:
            f.write("Bonus rows:\n")
            for r in bonus_rows:
                f.write(str(r) + "\n")

    print("Saved report notes:", path)


# =========================
# Program start
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_path = find_orl_folder(BASE_PATH)
    if dataset_path is None:
        sys.exit(1)

    print("Using ORL folder:", dataset_path)

    D, y = load_orl_dataset(dataset_path)

    # Basic images for the report
    save_sample_faces(D, y, os.path.join(OUTPUT_DIR, "sample_faces.png"))

    # Main 50% split
    X_train_50, y_train_50, X_test_50, y_test_50 = split_odd_even_rows(D, y)
    save_mean_and_eigenfaces(X_train_50, os.path.join(OUTPUT_DIR, "mean_face_and_eigenfaces.png"))
    rows_50 = run_pca_lda_experiments(X_train_50, y_train_50, X_test_50, y_test_50, "50_percent_5_train_5_test")

    # Bonus B: 7 train / 3 test split
    X_train_73, y_train_73, X_test_73, y_test_73 = split_7_train_3_test(D, y)
    rows_73 = run_pca_lda_experiments(X_train_73, y_train_73, X_test_73, y_test_73, "bonus_7_train_3_test")

    all_main_rows = rows_50 + rows_73
    print_table(all_main_rows, "Main + Bonus B Results")

    write_csv(
        all_main_rows,
        os.path.join(OUTPUT_DIR, "main_pca_lda_knn_and_7_3_split_results.csv"),
        ["Split", "Method", "Alpha", "K", "Components", "Accuracy"]
    )

    # Plots for 50% split
    plot_pca_alpha(rows_50, os.path.join(OUTPUT_DIR, "pca_alpha_relation_50_split.png"))

    # KNN plots: use the best PCA alpha from 50% split
    pca_rows_50 = [r for r in rows_50 if r["Method"] == "PCA"]
    best_pca_50 = max(pca_rows_50, key=lambda r: r["Accuracy"])
    plot_knn_for_method(rows_50, "PCA", os.path.join(OUTPUT_DIR, "pca_knn_tuning_50_split.png"), alpha=best_pca_50["Alpha"])
    plot_knn_for_method(rows_50, "LDA", os.path.join(OUTPUT_DIR, "lda_knn_tuning_50_split.png"))
    plot_split_comparison(rows_50, rows_73, os.path.join(OUTPUT_DIR, "split_50_vs_7_3_comparison.png"))

    # Save actual image examples, not only graphs
    best_pca = get_best_pca_predictions(X_train_50, y_train_50, X_test_50, y_test_50)
    save_success_failure_examples(
        X_test_50, y_test_50, best_pca["pred"],
        "PCA Success and Failure Cases",
        os.path.join(OUTPUT_DIR, "pca_success_failure_cases.png")
    )
    save_confusion_matrix(
        y_test_50, best_pca["pred"],
        "PCA Confusion Matrix",
        os.path.join(OUTPUT_DIR, "pca_confusion_matrix_40_subjects.png")
    )

    lda_best = get_lda_predictions(X_train_50, y_train_50, X_test_50, y_test_50, k=1)
    save_success_failure_examples(
        X_test_50, y_test_50, lda_best["pred"],
        "LDA Success and Failure Cases",
        os.path.join(OUTPUT_DIR, "lda_success_failure_cases.png")
    )
    save_confusion_matrix(
        y_test_50, lda_best["pred"],
        "LDA Confusion Matrix",
        os.path.join(OUTPUT_DIR, "lda_confusion_matrix_40_subjects.png")
    )

    # Bonus A: face vs non-face images
    non_faces = load_non_face_images(NON_FACE_PATH)
    bonus_rows = run_bonus_non_faces(D, non_faces)

    if bonus_rows:
        print_table(bonus_rows, "Bonus A Results: Faces vs Non-Faces")
        write_csv(
            bonus_rows,
            os.path.join(OUTPUT_DIR, "bonus_faces_vs_nonfaces_results.csv"),
            ["NonFaceTrainCount", "FixedFaceTrainCount", "PCA_Components", "PCA_Accuracy", "LDA_Components", "LDA_Accuracy"]
        )

    write_short_report_notes(rows_50, bonus_rows)

    print("\nDone.")
    print("Check the folder:", OUTPUT_DIR)
    print("Important output files:")
    print("- main_pca_lda_knn_and_7_3_split_results.csv")
    print("- pca_alpha_relation_50_split.png")
    print("- pca_knn_tuning_50_split.png")
    print("- lda_knn_tuning_50_split.png")
    print("- bonus_faces_vs_nonfaces_results.csv (only if non-face images exist)")
    print("- bonus_faces_vs_nonfaces_accuracy.png (only if non-face images exist)")
    print("- sample_faces.png")
    print("- mean_face_and_eigenfaces.png")
    print("- pca_success_failure_cases.png")
    print("- lda_success_failure_cases.png")
    print("- pca_confusion_matrix_40_subjects.png")
    print("- lda_confusion_matrix_40_subjects.png")
    print("- split_50_vs_7_3_comparison.png")
    print("- bonus_success_failure_cases.png (only if non-face images exist)")
    print("- report_notes.txt")


if __name__ == "__main__":
    main()
