import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
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
        """ Compute and store a normalized confusion matrix for the current run. """

        cm = confusion_matrix(y_true, y_pred)

        self.current_cm = cm
        self.history.append(cm)

    def compute_mean_std(self):

        normalized = []
        for cm in self.history:
            row_sums = cm.sum(axis=1, keepdims=True)
            # unikamy dzielenia przez 0
            row_sums[row_sums==0] = 1
            norm_cm = cm / row_sums * 100.0  # % 
            normalized.append(norm_cm)
        
        stacked = np.stack(normalized, axis=0)  # (num_folds, C, C)
        mean_cm = np.mean(stacked, axis=0)
        std_cm = np.std(stacked, axis=0)
        return mean_cm, std_cm

    def plot_cm(self, mode="mean", figsize=(8, 6), cmap="Blues"):
        """ Generate confusion matrix visualization. """

        fig, ax = plt.subplots(figsize=figsize)

        match mode:
            case "single":
            
                sns.heatmap(
                    self.current_cm,
                    annot=True,
                    fmt="",  
                    cmap=cmap,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar=True,
                    linewidths=1,
                    linecolor="white",
                    ax=ax
                )

                ax.set_title("Confusion Matrix")
        
            case "mean":

                mean_cm, std_cm = self.compute_mean_std()

                annotations = np.empty(mean_cm.shape, dtype=object)
                for i in range(mean_cm.shape[0]):
                    for j in range(mean_cm.shape[1]):
                        annotations[i,j] = f"{mean_cm[i,j]:.2f}%\n±{std_cm[i,j]:.2f}%"

                sns.heatmap(
                    mean_cm,
                    annot=annotations,
                    fmt="",
                    cmap=cmap,
                    vmin=0.0,
                    vmax=100.0,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar=True,
                    linewidths=1,
                    linecolor="white",
                    ax=ax
                )

                ax.set_title("Confusion Matrix (Mean ± Std, %)")
                
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        plt.tight_layout()
        return fig











        



  











    



            
    
    
    