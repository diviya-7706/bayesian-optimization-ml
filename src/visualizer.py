import matplotlib.pyplot as plt


def plot_convergence(bayes_scores, random_scores, save_path='plots/convergence_plot.png'):
    """
    Plots convergence curve comparing
    Bayesian Optimization vs Random Search.
    """
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
    plt.tight_layout()
    plt.savefig(save_path)
    return fig


def plot_accuracy_comparison(bayes_acc, random_acc, save_path='plots/accuracy_comparison.png'):
    """
    Plots bar chart comparing accuracy of
    Bayesian vs Random Search.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    methods = ['Bayesian Search', 'Random Search']
    scores = [bayes_acc, random_acc]
    colors = ['#1D9E75', '#D85A30']

    ax.bar(methods, scores, color=colors, width=0.4)
    ax.set_ylim(min(scores) - 0.02, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Method Comparison')

    for i, v in enumerate(scores):
        ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path)
    return fig
