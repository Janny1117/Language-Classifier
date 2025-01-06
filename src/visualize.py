import matplotlib.pyplot as plt

def visualize_benchmarking():
    """
    Generate benchmarking visualizations.
    """
    # Example F1 scores for benchmarking
    models = ['Baseline', 'Advanced Model 1', 'Advanced Model 2']
    f1_scores = [0.98, 0.99, 0.995]  # Replace with your actual scores

    # Plot benchmarking
    plt.bar(models, f1_scores, color=['blue', 'green', 'orange'])
    plt.ylabel('F1-Score')
    plt.title('Model Benchmarking')
    plt.savefig("evaluations/model_benchmark.png")

if __name__ == "__main__":
    visualize_benchmarking()