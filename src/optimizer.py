from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def run_bayesian_optimization(X_train, y_train, n_iter=20):
    """
    Runs Bayesian Optimization on Random Forest model.
    Returns the fitted optimizer object.
    """
    model = RandomForestClassifier(random_state=42)
    
    search_space = {
        'n_estimators': (10, 200),
        'max_depth': (1, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 10)
    }

    optimizer = BayesSearchCV(
        model,
        search_space,
        n_iter=n_iter,
        cv=5,
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    return optimizer


def run_random_search(X_train, y_train, n_iter=20):
    """
    Runs Random Search on Random Forest model.
    Used as baseline comparison against Bayesian Optimization.
    """
    model = RandomForestClassifier(random_state=42)

    param_dist = {
        'n_estimators': range(10, 200),
        'max_depth': range(1, 20),
        'min_samples_split': range(2, 10)
    }

    random_search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_iter,
        cv=5,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search


def get_improvement(bayes_score, random_score):
    """
    Calculates percentage improvement of
    Bayesian over Random Search.
    """
    return ((bayes_score - random_score) / random_score) * 100
