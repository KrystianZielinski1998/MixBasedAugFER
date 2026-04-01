import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class ClassificationReport:
    def __init__(self, class_names):
        self.class_names = class_names
        self.cr = None
        self.accuracy = 0.0

        self.cr_history = []  
        self.acc_history = []

    def update(self, all_labels, all_preds):
        cr = classification_report(
            all_labels,
            all_preds,
            output_dict=True,
            target_names=self.class_names
        )

        self.accuracy = cr.pop("accuracy")
        self.cr = cr

        self.cr_history.append(cr)
        self.acc_history.append(self.accuracy)

    def plot_cr(self, mode="single", figsize=(10,6), cmap="Blues"):
        
        match mode:
            case "single":
                report = self.cr
                acc = self.accuracy
                title = f"Single Report (Accuracy={acc:.4f})"

            case "mean":
                mean_dict, std_dict, acc_mean, acc_std = self.compute_mean_std()
                report = mean_dict
                title = f"Mean Report (Acc={acc_mean:.4f} ± {acc_std:.4f})"

        rows = list(report.keys())
        metrics = ["precision", "recall", "f1-score"]

        data = []
        annotations = []

        for r in rows:
            row_data = []
            row_annot = []

            for m in metrics:
                val = report[r][m]
                row_data.append(val)

                match mode:
                    case "mean":
                        std = std_dict[r][m]
                        row_annot.append(f"{val:.4f}\n±{std:.4f}")
                    case "single":
                        row_annot.append(f"{val:.4f}")

            data.append(row_data)
            annotations.append(row_annot)

        data = np.array(data)

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            data,
            annot=annotations,
            fmt="",
            cmap=cmap,
            cbar=True,
            linewidths=1,
            linecolor="white",
            xticklabels=metrics,
            yticklabels=rows,
            ax=ax
        )

        summary_start = len(rows) - 2
        ax.hlines(summary_start, *ax.get_xlim(), colors='white', linewidth=2.5, zorder=3)

        ax.set_title(title)
        ax.set_ylabel("Classes")
        ax.set_xlabel("Metrics")

        plt.tight_layout()

        return fig

    def compute_mean_std(self):
        if len(self.cr_history) == 0:
            raise ValueError("No reports in history.")

        rows = list(self.cr_history[0].keys())
        metrics = ["precision", "recall", "f1-score"]

        mean_dict = {}
        std_dict = {}

        for r in rows:
            mean_dict[r] = {}
            std_dict[r] = {}

            for m in metrics:
                values = [rep[r][m] for rep in self.cr_history]
                mean_dict[r][m] = np.mean(values)
                std_dict[r][m] = np.std(values)

        acc_mean = np.mean(self.acc_history)
        acc_std = np.std(self.acc_history)

        return mean_dict, std_dict, acc_mean, acc_std    











        



  











    



            
    
    
    