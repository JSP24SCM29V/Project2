import csv
import numpy as np
import pytest
from tabulate import tabulate
from model.GradientBoosting import GradientBoostingClassifier

def load_dataset(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data], dtype=np.float64)
    y = np.array([int(datum['y']) for datum in data], dtype=np.int64)
    return X, y

def print_model_metrics():
    datasets = [
        ("linear_separable.csv", 5),
        ("non_linear_moons.csv", 5),
        ("breast_cancer.csv", 5)
    ]
    table = []
    
    for dataset, max_depth in datasets:
        X, y = load_dataset(dataset)
        
        # Custom model
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=max_depth,
            min_samples_split=2,
            subsample=0.8,
            max_features=None
        )
        model.y_train_ = y  # Store training labels for Platt scaling
        results = model.fit(X, y)
        metrics = results.evaluate(X, y)
        
        # Add metrics to table
        table.append([
            dataset,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['log_loss']:.4f}"
        ])
    
    headers = ["Dataset", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Log Loss"]
    print("\nCustom Model Metrics Table:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

@pytest.mark.parametrize("dataset, min_accuracy, min_roc_auc, max_log_loss, max_depth", [
    ("linear_separable.csv", 0.95, 0.95, 0.2, 5),
    ("non_linear_moons.csv", 0.95, 0.95, 0.2, 5),
    ("breast_cancer.csv", 0.95, 0.95, 0.2, 5)
])
def test_model_performance(dataset, min_accuracy, min_roc_auc, max_log_loss, max_depth):
    X, y = load_dataset(dataset)
    
    # Test custom model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=max_depth,
        min_samples_split=2,
        subsample=0.8,
        max_features=None
    )
    model.y_train_ = y
    results = model.fit(X, y)
    metrics = results.evaluate(X, y)

    assert metrics['accuracy'] >= min_accuracy, f"Accuracy too low for {dataset}: {metrics['accuracy']}"
    assert metrics['precision'] >= min_accuracy * 0.95, f"Precision too low for {dataset}: {metrics['precision']}"
    assert metrics['recall'] >= min_accuracy * 0.95, f"Recall too low for {dataset}: {metrics['recall']}"
    assert metrics['f1_score'] >= min_accuracy * 0.95, f"F1-score too low for {dataset}: {metrics['f1_score']}"
    assert metrics['roc_auc'] >= min_roc_auc, f"ROC AUC too low for {dataset}: {metrics['roc_auc']}"
    assert metrics['log_loss'] <= max_log_loss, f"Log loss too high for {dataset}: {metrics['log_loss']}"

def test_predict_proba():
    X, y = load_dataset("linear_separable.csv")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=2,
        subsample=0.8,
        max_features=None
    )
    model.y_train_ = y
    results = model.fit(X, y)
    proba = results.predict_proba(X)

    assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of range"
    assert np.allclose(proba.sum(axis=1), 1), "Probabilities do not sum to 1"

def test_print_metrics():
    print_model_metrics()

