# # Bayesian Optimization for ML Models 

##  Live Demo
(https://huggingface.co/spaces/diviya7706/bayesian-optimization)
---

## About the Project

This project demonstrates the use of **Bayesian Optimization** to automatically 
tune machine learning model hyperparameters. Instead of trying every possible 
combination (grid search) or random combinations (random search), Bayesian 
Optimization intelligently learns from each trial and picks the next best 
combination — achieving higher accuracy in fewer attempts.

---

##  Model Results

| Method | Accuracy | Trials |
|--------|----------|--------|
| Bayesian Optimization | **0.9583** | 20 |
| Random Search | 0.9500 | 20 |
| Improvement | **+0.88%** | — |

> Tested on Iris Dataset — 150 samples, 4 features, 3 classes

---


## Features

- Choose from multiple datasets (Breast Cancer, Iris, Wine)
- Upload your own CSV dataset
- Adjustable number of optimization trials
- Live convergence curve comparing Bayesian vs Random Search
- Best hyperparameters displayed automatically
- Accuracy comparison bar chart
- Fully deployed and accessible online

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core programming language |
| Streamlit | Web app framework |
| scikit-optimize | Bayesian Optimization library |
| scikit-learn | ML models and datasets |
| matplotlib | Charts and visualizations |
| seaborn | Statistical plots |
| pandas | Data manipulation |
| numpy | Numerical computing |

---

## Project Structure
bayesian-optimization-ml/
- │
- ├── app.py                  ← Main Streamlit application
- ├── requirements.txt        ← Python dependencies
- ├── README.md               ← Project documentation
- │
- ├── notebooks/
- │   └── Bayesian_Optimization_ML.ipynb  ← Experiments
- │
- ├── plots/
- │   ├── convergence_plot.png            ← Results chart
- │   └── accuracy_comparison.png
- │
- └── data/
- └── your_datasets.csv

---

##  Installation & Run Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/diviya7706/bayesian-optimization-ml.git
cd bayesian-optimization-ml
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run app.py
```

**Step 4 — Open in browser**
- http://localhost:8501

---

## Requirements
- matplotlib
- numpy
- pandas
- seaborn
- scikit-learn
- scikit-optimize
- streamlit

---

## How Bayesian Optimization Works

1. **Start** with a few random hyperparameter trials
2. **Build** a surrogate model (Gaussian Process) from results
3. **Predict** which combination will perform best next
4. **Test** that combination on the actual model
5. **Update** the surrogate model with new results
6. **Repeat** until the best parameters are found

This is smarter and faster than trying all combinations manually.

---

## Hyperparameters Tuned

| Parameter | Search Range | Best Value Found |
|-----------|-------------|-----------------|
| n_estimators | 10 — 200 | ~150 |
| max_depth | 1 — 20 | ~10 |
| min_samples_split | 2 — 10 | ~3 |
| min_samples_leaf | 1 — 10 | ~1 |

---

## Real World Applications

- Drug discovery and medical research
- Neural network architecture search
- Chip design optimization
- Robotics and control systems
- Financial trading strategy tuning
- Climate and energy optimization

---

## Author

**Diviya Dharshini**
- Hugging Face: [diviya7706](https://huggingface.co/diviya7706)
- GitHub: [diviya7706](https://github.com/diviya7706)

---

## License

This project is open source and available under the [MIT License](LICENSE).
