import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(df):
    df["Class"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig("outputs/plots/class_distribution.png")
    plt.close()


def plot_model_comparison(df):
    df.set_index("Model").plot(kind="bar", figsize=(8, 5))
    plt.ylim(0, 1)
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig("outputs/plots/model_comparison.png")
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, title, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{filename}.png")
    plt.close()


def plot_roc(fpr, tpr, auc_score):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/roc.png")
    plt.close()
