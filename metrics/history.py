import seaborn as sns
import matplotlib.pyplot as plt

class History:
    def __init__(self):
        pass

    def plot_his(self, metric="loss", figsize=(10, 6)):
    
        fig, ax = plt.subplots(figsize=figsize)

        match metric:
            case "loss":

                train = self.history["train_loss"]
                val = self.history["val_loss"]
                ylabel = "Val/Train Loss"
        
        match metric: 
            case "acc":

                train = self.history["train_acc"]
                val = self.history["val_acc"]
                ylabel = "Val/Train Accuracy"
        
        epochs = range(1, len(train) + 1)

        sns.lineplot(x=epochs, y=train, label="Train", ax=ax)
        sns.lineplot(x=epochs, y=val, label="Validation", ax=ax)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}")
        ax.legend()

        plt.tight_layout()
        plt.show()

        return fig











        



  











    



            
    
    
    