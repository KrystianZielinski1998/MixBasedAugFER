import wandb 

class WandbLogger:
    def __init__(self):
        pass

    def log_fig(self, fig, name="confusion_matrix"):
        """ Log figure. """

        wandb.log({name: wandb.Image(fig)})


    
    

    

        

        
        
        

    




            
    
    
    