import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimizer import run_bayesian_optimization, run_random_search, get_improvement
from src.data_loader import load_dataset, split_data


def test_bayesian_optimization():
    """Test that Bayesian optimization runs and returns valid accuracy."""
    X, y = load_dataset("Iris")
    X_train, X_test, y_train, y_test = split_data(X, y)
    optimizer = run_bayesian_optimization(X_train, y_train, n_iter=5)
    assert optimizer.best_score_ > 0.5, "Accuracy should be above 50%"
    print(f"✅ Bayesian Optimization passed — Accuracy: {optimizer.best_score_:.4f}")


def test_random_search():
    """Test that Random Search runs and returns valid accuracy."""
    X, y = load_dataset("Iris")
    X_train, X_test, y_train, y_test = split_data(X, y)
    random_search = run_random_search(X_train, y_train, n_iter=5)
    assert random_search.best_score_ > 0.5, "Accuracy should be above 50%"
    print(f"✅ Random Search passed — Accuracy: {random_search.best_score_:.4f}")


def test_improvement_calculation():
    """Test improvement calculation is correct."""
    improvement = get_improvement(0.95, 0.90)
    assert round(improvement, 2) == 5.56, "Improvement calculation is wrong"
    print(f"✅ Improvement calculation passed — {improvement:.2f}%")


def test_data_loader():
    """Test that datasets load correctly."""
    for name in ["Breast Cancer", "Iris", "Wine"]:
        X, y = load_dataset(name)
        assert X.shape[0] > 0, f"{name} dataset is empty"
        print(f"✅ {name} dataset loaded — {X.shape[0]} samples")


if __name__ == "__main__":
    test_data_loader()
    test_bayesian_optimization()
    test_random_search()
    test_improvement_calculation()
    print("\n🎉 All tests passed!")
