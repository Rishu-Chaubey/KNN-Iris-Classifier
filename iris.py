import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# -----------------------------
# Utility: ensure output folders
# -----------------------------
def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


# ------------------------------------
# 1) Load dataset and basic inspection
# ------------------------------------
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Add numeric target and a human-readable label column
    df["target"] = iris.target
    df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, iris


# -------------------------
# 2) Exploratory Data Analysis
# -------------------------
def run_eda(df):
    print("=== Basic Info ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nHead:\n", df.head(), "\n")

    # Missing values
    print("=== Missing Values Per Column ===")
    print(df.isnull().sum(), "\n")

    # Class distribution plot (by label for readability)
    ax = df["target_name"].value_counts().sort_index().plot(kind="bar", color="#4c72b0")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("figures/class_distribution.png", dpi=150)
    plt.close()

    # Pairplot colored by species label
    sns.pairplot(df.drop(columns=["target"]), hue="target_name", corner=True, diag_kind="hist")
    plt.tight_layout()
    plt.savefig("figures/pairplot.png", dpi=150)
    plt.close()

    # Correlation heatmap (only numeric features)
    numeric_cols = [c for c in df.columns if c not in ["target", "target_name"]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png", dpi=150)
    plt.close()

    # Outlier detection via Z-score (|z| > 3)
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    outlier_counts = (z_scores > 3).sum()
    print("=== Potential Outliers (|z| > 3) Per Feature ===")
    print(outlier_counts.to_string(), "\n")


# ------------------------------------
# 3) Preprocess, split, and scale data
# ------------------------------------
def preprocess_and_split(df, test_size=0.30, random_state=42):
    X = df.drop(columns=["target", "target_name"])
    y = df["target"]

    # Stratify to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Standard scaling (fit on train, apply to train and test)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler


# -------------------------
# 4) Train and evaluate KNN
# -------------------------
def train_and_evaluate(X_train_s, X_test_s, y_train, y_test, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_s, y_train)

    y_pred = knn.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=3)

    print("=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    return knn, acc, cm, cr


# -------------------------
# 5) Persist artifacts
# -------------------------
def save_artifacts(model, scaler, model_path="models/knn_iris_model.pkl", scaler_path="models/scaler.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to:   {model_path}")
    print(f"Saved scaler to:  {scaler_path}")


# -------------------------
# Main execution pipeline
# -------------------------
def main():
    ensure_dirs()

    # Load and EDA
    df, iris = load_data()
    run_eda(df)

    # Split and scale
    X_train_s, X_test_s, y_train, y_test, scaler = preprocess_and_split(df)

    # Train and evaluate
    knn, acc, cm, cr = train_and_evaluate(X_train_s, X_test_s, y_train, y_test, n_neighbors=3)

    # Save artifacts for Streamlit app
    save_artifacts(knn, scaler)


if __name__ == "__main__":
    main()
