from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_dataset(name):
    """
    Loads a built-in sklearn dataset by name.
    Returns X, y arrays.
    """
    datasets = {
        "Breast Cancer": load_breast_cancer,
        "Iris": load_iris,
        "Wine": load_wine
    }

    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found.")

    data = datasets[name]()
    return data.data, data.target


def load_csv_dataset(filepath, target_column):
    """
    Loads a custom CSV dataset.
    Returns X, y arrays.
    """
    df = pd.read_csv(filepath)
    X = df.drop(columns=[target_column]).values
    y = LabelEncoder().fit_transform(df[target_column])
    return X, y


def split_data(X, y, test_size=0.2):
    """
    Splits data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)
