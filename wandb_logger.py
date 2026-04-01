import wandb 

class WandbLogger:
    def __init__(self):
        pass

    def log_history(self, history):
        """ 
        Log full training history as a W&B table.
        
        Args:
            history (dict): {
                "train_loss": [...],
                "train_acc": [...],
                "val_loss": [...],
                "val_acc": [...]
            }
        """
        
        # Determine number of epochs
        num_epochs = len(history["train_loss"])

        # Create table
        table = wandb.Table(columns=[
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])

        # Fill table
        for epoch in range(num_epochs):
            table.add_data(
                epoch,
                history["train_loss"][epoch],
                history["train_acc"][epoch],
                history["val_loss"][epoch],
                history["val_acc"][epoch]
            )

        # Log table
        wandb.log({"training_history": table})

    def log_confusion_matrix(self, all_labels, all_preds, class_names=None):
        """ Log confusion matrix. """
    
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )
        })
    
    def log_confusion_matrix_fig(self, fig):
        """ Log confusion matrix figure. """

        wandb.log({"confusion_matrix": wandb.Image(fig)})

    def log_classification_report_fig(self, fig):
        """ Log classification report figure. """

        wandb.log({"classification_report": wandb.Image(fig)})
    
    

    

        

        
        
        

    




            
    
    
    