import wandb 

class WandbLogger:
    def __init__(self):
        pass

    def log_fig(self, fig, name):
        """ Log figure. """

        wandb.log({name: wandb.Image(fig)})

    def log_artifact(self, model_path, artifact_name):
        """ Log model artifact. """

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


    
    

    

        

        
        
        

    




            
    
    
    