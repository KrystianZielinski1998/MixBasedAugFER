import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class ConfusionMatrixTracker:
    """
    Utility class for tracking, aggregating, and visualizing confusion matrices.

    The class stores normalized confusion matrices (row-wise) and provides
    functionality to compute mean and standard deviation, as well as to
    visualize the aggregated result.
    """

    def __init__(self, class_names):
        """
        Initialize the tracker.

        Args:
            class_names (list of str): Names of the classes.
        """
        self.class_names = class_names
        self.current_cm = None
        self.history = []  # List of normalized confusion matrices

    def update(self, y_true, y_pred):
        """
        Compute and store a normalized confusion matrix for the current run.

        Normalization is performed row-wise (i.e., over true labels), so each
        row sums to 1. This ensures comparability across runs with different
        sample sizes.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
        """
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        self.current_cm = cm
        self.history.append(cm)

    def compute_mean_std(self):
        """
        Compute the mean and standard deviation of stored confusion matrices.

        Returns:
            tuple:
                mean_cm (ndarray): Mean confusion matrix.
                std_cm (ndarray): Standard deviation matrix.

        Raises:
            ValueError: If no matrices are stored.
        """
        if len(self.history) == 0:
            raise ValueError("No confusion matrices in history.")

        stacked = np.stack(self.history, axis=0)  # Shape: (N, C, C)

        mean_cm = np.mean(stacked, axis=0)
        std_cm = np.std(stacked, axis=0)

        return mean_cm, std_cm

    def plot_cm(self, figsize=(8, 6), cmap="Blues"):
        """
        Generate a heatmap visualization of the mean confusion matrix with
        standard deviation annotations.

        Values are displayed as percentages (mean ± std).

        Args:
            figsize (tuple): Figure size.
            cmap (str): Colormap for the heatmap.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        mean_cm, std_cm = self.compute_mean_std()

        # Prepare annotation strings (percentage format)
        annotations = np.empty_like(mean_cm).astype(str)

        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                mean_val = mean_cm[i, j] * 100.0
                std_val = std_cm[i, j] * 100.0

                annotations[i, j] = f"{mean_val:.2f}%\n±{std_val:.2f}%"

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            mean_cm,
            annot=annotations,
            fmt="",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True,
            linewidths=1,
            linecolor="white",
            ax=ax
        )

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix (Mean ± Std, %)")

        plt.tight_layout()

        return fig











        



  











    



            
    
    
    