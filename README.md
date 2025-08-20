# KNN Iris Classifier — Streamlit App

Predicts Iris flower species (Setosa, Versicolor, Virginica) from four numeric measurements:
- Sepal length
- Sepal width
- Petal length
- Petal width

This repository shows the full path from training a KNN model to deploying a small, interactive web app using Streamlit.

## What’s in this project and why it exists

- Goal: Demonstrate a simple, reproducible ML workflow end‑to‑end.
- Why KNN: It’s an intuitive classifier that works well on the classic Iris dataset and is easy to explain.
- Why Streamlit: It lets anyone try the model in a browser without writing code.

## How the pieces fit together (high level)

1. iris.py trains the model
   - Loads the Iris dataset
   - Splits into train/test
   - Scales features (important for distance‑based models like KNN)
   - Trains KNN (k=3 by default)
   - Evaluates performance (accuracy, confusion matrix, classification report)
   - Saves two artifacts into models/:
     - knn_iris_model.pkl (the trained classifier)
     - scaler.pkl (the fitted StandardScaler used at training time)

2. app.py serves predictions
   - Builds a Streamlit UI with sliders for the four features
   - Loads the saved scaler and model from models/
   - Transforms user inputs with the same scaler used during training
   - Runs model.predict(...) and shows the predicted species with color/emoji
   - Offers a button or section to display metrics and info in the sidebar

This separation keeps training and inference clean and maintainable.

## Quickstart (run the app locally)

1) Clone
```bash
git clone https://github.com/Rishu-Chaubey/knn-iris-classifier.git
cd knn-iris-classifier
```

2) Install dependencies
```bash
pip install -r requirements.txt
```
- Why: Ensures the same library versions used during development are available on the machine running the app.

3) Run the Streamlit app
```bash
streamlit run app.py
```
- Streamlit will print a local URL (usually http://localhost:8501). Open it in a browser.
- Move the sliders to set the four measurements and observe the predicted species.

## Project structure (with comments)

```
knn-iris-classifier/
├─ models/
│  ├─ knn_iris_model.pkl    # Trained KNN model saved after training (binary file)
│  └─ scaler.pkl            # Fitted StandardScaler to apply identical scaling at inference
├─ app.py                   # Streamlit UI + loads artifacts + makes predictions
├─ iris.py                  # Training pipeline: load data → preprocess → train → evaluate → save
├─ requirements.txt         # List of dependencies to reproduce the environment
├─ LICENSE                  # MIT License (you are free to use/modify with attribution)
└─ README.md                # Project documentation (this file)
```

## Detailed walkthrough

1. Data and preprocessing
   - Dataset: The classic Iris dataset (150 rows, 3 classes, 4 features).
   - Scaling: StandardScaler transforms each feature to mean 0, std 1.
     - Why it matters: KNN uses distances between points; unscaled features can dominate the distance if they have larger numeric ranges.
   - Train/test split: Keeps a portion of data unseen during training to fairly measure performance.

2. Model choice: K‑Nearest Neighbors (KNN)
   - Idea: For a new sample, find the k closest training samples and vote on the label.
   - Hyperparameter k: Common small values are 3, 5, 7. This project defaults to k=3.
   - Pros: Simple, interpretable; no heavy training stage.
   - Cons: Prediction can be slower on very large datasets (distance checks), sensitive to scaling.

3. Saving artifacts
   - Why save scaler.pkl and knn_iris_model.pkl?
     - The app must apply the exact same preprocessing (scaling) as training.
     - The model is reused without retraining, making deployment instant.

4. Streamlit app behavior
   - UI: Sliders for sepal/petal measurements.
   - On submit/change:
     - Collect values → build a 2D array (shape: 1×4) → transform with scaler → predict with KNN.
   - Output: Species name with a small color highlight and an emoji to improve readability.
   - Sidebar: Quick facts (model type, k value, train/test accuracy). Optional button to show the full classification report.

## Retraining or modifying the model

- Change k: Open iris.py and update n_neighbors in KNeighborsClassifier.
- Re‑train:
  ```bash
  python iris.py
  ```
  This regenerates the model files under models/.

- Swap model type (e.g., SVM, RandomForest):
  - Replace the classifier in iris.py.
  - Keep the scaling step if needed for the chosen model.
  - Re‑save artifacts and ensure app.py still loads them properly.

## Common issues and tips

- Module or version not found
  - Run pip install -r requirements.txt again.
  - Ensure the active Python environment matches where you installed packages.

- File not found: models/scaler.pkl or models/knn_iris_model.pkl
  - Make sure models/ contains both files.
  - If missing, run python iris.py to recreate them.

- Incorrect predictions after code tweaks
  - If you change preprocessing or k, retrain and overwrite both artifacts so the app and training match.

## Tech stack justification

- scikit‑learn: Reliable, battle‑tested ML utilities and models.
- Streamlit: Zero‑boilerplate web UI for data apps; great for demos and quick deployment.
- NumPy/Pandas: Fast numeric and tabular manipulations.
- Seaborn/Matplotlib: EDA plots for better understanding of the data.
- Pickle/Joblib: Simple, standard ways to persist models and preprocessors.

## License

MIT License — permissive and simple. You may use, modify, and distribute this code with attribution. See LICENSE for the full text.

## Acknowledgments

- Iris dataset (Fisher/UC Irvine; bundled with scikit‑learn)
- Streamlit for effortless UI
- scikit‑learn for ML components
