import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import logging

class GetLoaders:
    def __init__(self, dataset_path, batch_size=64):
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size

    def get_transforms(self):
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        val_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return train_tf, val_tf

    def __call__(self, fold_idx):
        train_tf, val_tf = self.get_transforms()

        train_path = self.dataset_path / f"fold_{fold_idx}" / "train"
        val_path   = self.dataset_path / f"fold_{fold_idx}" / "validation"
        test_path  = self.dataset_path / f"fold_{fold_idx}" / "test"

        train_ds = datasets.ImageFolder(train_path, transform=train_tf)
        val_ds = datasets.ImageFolder(val_path, transform=val_tf)
        test_ds = datasets.ImageFolder(test_path, transform=val_tf)

        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.batch_size),
            DataLoader(test_ds, batch_size=self.batch_size),
        )

class EarlyStopping:
    """ Early stopping to stop training when validation accuracy doesn't improve. Saves the best checkpoint. """

    def __init__(self, 
        patience: int, 
        min_delta: float,
        verbose: bool
    ):
        """   
        Args:
            patience (int): Number of epochs to wait after min has been hit before stopping.
            min_delta (float): Minimum change in monitored value to qualify as improvement.
            verbose (bool): Add logs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

        self.logger = logging.getLogger(__name__)

        
    def __call__(self, val_acc, model, fold_idx):
        if val_acc > self.best_acc + self.min_delta:
            if self.verbose:
                self.logger.info(f'Accuracy on validation set increased from {self.best_acc: .2f} to {val_acc: .2f}. Saving model...')
            self.best_acc = val_acc
            self.counter = 0
            
            # Save the model
            torch.save(model.state_dict(), f"best_model_fold_{fold_idx}.pth")
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'Early stopping counter: {self.counter} out of {self.patience}.')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info('Early stopping triggered.')

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        max_epochs,
        patience,
        base_lr,
        patience_lr,
        min_lr,
        factor_lr,
        fold_idx
    ):
        self.model = model.to(device)
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.max_epochs = max_epochs
        self.patience = patience
        self.base_lr = base_lr
        self.patience_lr = patience_lr
        self.min_lr = min_lr
        self.factor_lr = factor_lr

        self.fold_idx = fold_idx

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',               
            factor=self.factor_lr,               
            patience=self.patience_lr,
            min_lr=self.min_lr
        )    

        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0, verbose=True)

        self.history = defaultdict(list)

        self.logger = logging.getLogger(__name__) 

    # -------------------
    # TRAIN ONE EPOCH
    # -------------------
    def train_one_epoch(self):
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, 100 * correct / total

    # -------------------
    # EVALUATION
    # -------------------
    def evaluate(self):
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * imgs.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, 100 * correct / total

    # -------------------
    # TEST
    # -------------------
    def test(self):
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)
                preds = outputs.argmax(dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        return all_labels, all_preds

    # -------------------
    # TRAIN LOOP
    # -------------------
    def __call__(self):

        for epoch in range(self.max_epochs):
            self.logger.info(f"___________________________")
            self.logger.info(f"Epoch: {epoch+1} / {self.max_epochs}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Current LR: {current_lr:.6f}")  

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()

            # Save metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}")
            self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}")

            # Early stopping 
            self.early_stopping(val_acc, self.model, self.fold_idx)
            if self.early_stopping.early_stop:
                break

            # Scheduler
            self.scheduler.step(val_acc)

        self.logger.info(f"___________________________")
        self.logger.info(f"Evaluating on t est set.")

        # Evaluate on test set - get accuracy, confusion matrix and classification report with all the metrics
        all_labels, all_preds = self.test()

        return self.history, all_labels, all_preds

        







        



  











    



            
    
    
    