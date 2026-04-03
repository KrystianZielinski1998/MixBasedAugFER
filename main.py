import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
import wandb
from torchvision.models import (
    convnext_base,
    shufflenet_v2_x2_0,
    swin_b,
    ConvNeXt_Base_Weights,
    ShuffleNet_V2_X2_0_Weights,
    Swin_B_Weights
)

from train import GetLoaders, Trainer
from utils.logging_config import setup_logging
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_report import ClassificationReport
from metrics.history import History
from wandb_logger import WandbLogger

def get_model(model_str: str, classes: list):

    num_classes = len(classes)
    match model_str:
        case "convnext":
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 
            model = convnext_base(weights=weights)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        case "shufflenet":
            weights = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
            model = shufflenet_v2_x2_0(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        case "swin":
            weights = Swin_B_Weights.IMAGENET1K_V1
            model = swin_b(weights=weights)
            model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model, weights

def parse_args():
    """
    Parse command-line arguments for training and logging configuration.
    Returns a namespace with all arguments.
    """

    parser = argparse.ArgumentParser(
        description="Training script with W&B logging and k-fold support"
    )

    # Dataset parameters
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset")
    
    # Model parameters
    parser.add_argument("--model_str", type=str, default="shufflenet", choices=["convnext", "shufflenet", "swin"], help="Model architecture to use")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--base_lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--patience_lr", type=int, default=5, help="LR scheduler patience")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--factor_lr", type=float, default=1e-2, help="Factor for learning rate scheduler")

    # K-fold parameters
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross-validation")

    # Augmentation method
    parser.add_argument("--aug_method", type=str, default="none", help="Data augmentation method applied")
  
    args = parser.parse_args()

    # Get dataset path
    args.dataset_path = args.dataset_name + " 5-fold CV"

    # Convert dataset path to Path object
    args.dataset_path = Path(args.dataset_path)

    return args

def main():

    # Parse args
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class names list
    classes = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    # Get loaders
    get_loaders = GetLoaders(args.dataset_path, args.batch_size)

    # Initialize metric classes
    cr = ClassificationReport(classes)
    cm = ConfusionMatrix(classes)
    his = History()

    # Initialize wandblogger
    wandb_logger = WandbLogger()

    # Logger
    logger = logging.getLogger(__name__) 

    # Group name
    group_name = f"{args.model_str}_{args.dataset_name}_{args.aug_method}"

    # Start k-fold cross validation
    for fold in range(1, args.num_folds + 1):
        
        # Get model 
        model, weights = get_model(args.model_str, classes)

        # Get training, validation and test dataset loaders
        train_loader, val_loader, test_loader = get_loaders(
          fold_idx=fold
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            base_lr=args.base_lr,
            patience_lr=args.patience_lr,
            min_lr=args.min_lr,
            factor_lr=args.factor_lr,
            fold_idx=fold
        )

        # Initialize wandb for logging metrics
        wandb.init(
            project="DataAugmentation",
            group=group_name,
            name=f"fold_{fold}",
            config={
                "fold": fold,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "base_lr": args.base_lr,
                "patience_lr": args.patience_lr,
                "min_lr": args.min_lr
                "factor_lr": args.factor_lr
            }
        )

        # Start training
        history, all_labels, all_preds = trainer()

        # Update classification report and confusion matrix
        cm.update(all_labels, all_preds)
        cr.update(all_labels, all_preds)

        # Draw Train/Val Accuracy and Loss line plots
        his_fig_acc = his.plot_his(history, metric="acc")
        his_fig_loss = his.plot_his(history, metric="loss")

        # Log Train/Val Accuracy and Loss line plots
        wandb_logger.log_fig(his_fig_acc, "train/val_acc")
        wandb_logger.log_fig(his_fig_loss, "train/val_loss")

        # Draw heatmap plot for classification report
        cm_fig = cm.plot_cm(mode="single")
        cr_fig = cr.plot_cr(mode="single")

        # Log classification report and confusion matrix
        wandb_logger.log_fig(cm_fig, "confusion_matrix")
        wandb_logger.log_fig(cr_fig, "classification_report")

        # Log model artifact 
        wandb_logger.log_artifact(f"best_model_fold_{fold}.pth", f"{group_name}_{fold}")

        # Finish logging for current fold
        wandb.finish()

    wandb.init(
            project="DataAugmentation",
            group=group_name,
            name=f"mean",
            config={
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "base_lr": args.base_lr,
                "patience_lr": args.patience_lr,
                "min_lr": args.min_lr,
                "factor_lr": args.factor_lr
            }
        )

    # Plot mean confusion matrix and classification report
    cm_mean_fig = cm.plot_cm(mode="mean")
    cr_mean_fig = cr.plot_cr(mode="mean")

    # Log mean confusion matrix and classification report
    wandb_logger.log_fig(cm_mean_fig, "confusion_matrix")
    wandb_logger.log_fig(cr_mean_fig, "classification_report")

    # Finish logging for averaging results from k-fold CV
    wandb.finish()

if __name__ == "__main__":
  setup_logging()
  main()




        








    



    



            
    
    
    