import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bayesian Optimization", layout="wide")

st.title("Bayesian Optimization for ML Models")
st.markdown("Optimize machine learning hyperparameters using Bayesian techniques.")

st.sidebar.header("Settings")

dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["Breast Cancer", "Iris"]
)

n_iter = st.sidebar.slider("Number of Trials", min_value=10, max_value=50, value=20)

st.sidebar.markdown("---")
st.sidebar.info("More trials = better accuracy but takes longer.")

if dataset_choice == "Breast Cancer":
    data = load_breast_cancer()
else:
    data = load_iris()

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.markdown(f"### Dataset: {dataset_choice}")
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", X.shape[0])
col2.metric("Features", X.shape[1])
col3.metric("Classes", len(np.unique(y)))

st.markdown("---")

if st.button("Run Bayesian Optimization"):

    with st.spinner("Running optimization... please wait..."):


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

    
        from sklearn.model_selection import RandomizedSearchCV
        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            {
                'n_estimators': range(10, 200),
                'max_depth': range(1, 20),
                'min_samples_split': range(2, 10)
            },
            n_iter=n_iter,
            cv=5,
            random_state=42
        )
        random_search.fit(X_train, y_train)

    st.success("Optimization complete!")

  
    st.markdown("### Results")

    bayes_acc = optimizer.best_score_
    random_acc = random_search.best_score_
    improvement = ((bayes_acc - random_acc) / random_acc) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Bayesian Best Accuracy", f"{bayes_acc:.4f}")
    c2.metric("Random Search Accuracy", f"{random_acc:.4f}")
    c3.metric("Improvement", f"{improvement:.2f}%")

  
    st.markdown("### Best Hyperparameters Found")
    params_df = pd.DataFrame([optimizer.best_params_])
    st.dataframe(params_df, use_container_width=True)

    st.markdown("### Convergence Curve — Bayesian vs Random Search")

    bayes_scores = optimizer.cv_results_['mean_test_score']
    random_scores = random_search.cv_results_['mean_test_score']

    bayes_best = [max(bayes_scores[:i+1]) for i in range(len(bayes_scores))]
    random_best = [max(random_scores[:i+1]) for i in range(len(random_scores))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bayes_best, label='Bayesian Search', color='#1D9E75', linewidth=2)
    ax.plot(random_best, label='Random Search', color='#D85A30', linewidth=2, linestyle='--')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Best Accuracy So Far')
    ax.set_title('Optimization Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.savefig('convergence_plot.png')

    st.markdown("### Accuracy Comparison")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    methods = ['Bayesian Search', 'Random Search']
    scores = [bayes_acc, random_acc]
    colors = ['#1D9E75', '#D85A30']
    ax2.bar(methods, scores, color=colors, width=0.4)
    ax2.set_ylim(min(scores) - 0.02, 1.0)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Method Comparison')
    for i, v in enumerate(scores):
        ax2.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=11)
    st.pyplot(fig2)

    st.markdown("---")
    st.markdown("**Plots saved as** `convergence_plot.png` in your project folder.")
