import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, threshold=0.5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.threshold = threshold
        
        # Initialize metrics storage
        self.metrics = {
            'train_metrics': {
                'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []
            },
            'val_metrics': {
                'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []
            }
        }
        
        # Initialize metric modules
        self.train_metrics = nn.ModuleDict({
            'acc': Accuracy(task='binary').to(device),
            'prec': Precision(task='binary').to(device),
            'rec': Recall(task='binary').to(device),
            'f1': F1Score(task='binary').to(device)
        })
        
        self.val_metrics = nn.ModuleDict({
            'acc': Accuracy(task='binary').to(device),
            'prec': Precision(task='binary').to(device),
            'rec': Recall(task='binary').to(device),
            'f1': F1Score(task='binary').to(device)
        })
        
        # Loss and optimizer
        self.criterion = WeightedFocalLoss(alpha=50, gamma=2)
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=3, factor=0.5)

    def _compute_metrics(self, preds, targets, mode='train'):
        """Calculate batch metrics"""
        with torch.no_grad():
            bin_preds = (preds > self.threshold).float()
            metrics = self.train_metrics if mode == 'train' else self.val_metrics
            
            acc = metrics['acc'](bin_preds, targets)
            prec = metrics['prec'](bin_preds, targets)
            rec = metrics['rec'](bin_preds, targets)
            f1 = metrics['f1'](bin_preds, targets)
            
        return acc, prec, rec, f1

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for metric in self.train_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()

        # Compute epoch metrics
        with torch.no_grad():
            self.model.eval()
            acc, prec, rec, f1 = self._compute_metrics(outputs, targets, 'train')
            self.model.train()
            
        # Store metrics
        n = len(self.train_loader)
        self.metrics['train_metrics']['loss'].append(epoch_loss/n)
        self.metrics['train_metrics']['acc'].append(acc.item())
        self.metrics['train_metrics']['prec'].append(prec.item())
        self.metrics['train_metrics']['rec'].append(rec.item())
        self.metrics['train_metrics']['f1'].append(f1.item())

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_loss = 0
        for metric in self.val_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            epoch_loss += loss.item()
            
            # Update metrics
            self.val_metrics['acc'].update(outputs, targets)
            self.val_metrics['prec'].update(outputs, targets)
            self.val_metrics['rec'].update(outputs, targets)
            self.val_metrics['f1'].update(outputs, targets)

        # Compute final metrics
        final_acc = self.val_metrics['acc'].compute().item()
        final_prec = self.val_metrics['prec'].compute().item()
        final_rec = self.val_metrics['rec'].compute().item()
        final_f1 = self.val_metrics['f1'].compute().item()

        # Store metrics
        n = len(self.val_loader)
        self.metrics['val_metrics']['loss'].append(epoch_loss/n)
        self.metrics['val_metrics']['acc'].append(final_acc)
        self.metrics['val_metrics']['prec'].append(final_prec)
        self.metrics['val_metrics']['rec'].append(final_rec)
        self.metrics['val_metrics']['f1'].append(final_f1)

        return final_f1

    def fit(self, epochs, early_stop=5):
        best_f1 = 0
        no_improve = 0
        
        for epoch in range(epochs):
            self.train_epoch()
            val_f1 = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_f1)
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improve += 1
                
            if no_improve >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            self._print_metrics(epoch+1)

    def _print_metrics(self, epoch):
    #"""Print metrics for current epoch"""
        train = self.metrics['train_metrics']  # Changed from 'train'
        val = self.metrics['val_metrics']      # Changed from 'val'

        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train['loss'][-1]:.4f} | Acc: {train['acc'][-1]:.4f} | F1: {train['f1'][-1]:.4f}")
        print(f"Val   Loss: {val['loss'][-1]:.4f} | Acc: {val['acc'][-1]:.4f} | F1: {val['f1'][-1]:.4f}\n")



        
    def _compute_metrics(self, preds, targets, mode='train'):
        #"""Calculate batch metrics"""
        with torch.no_grad():
            bin_preds = (preds > self.threshold).float()
            
            if mode == 'train':
                metrics = self.train_metrics
            else:
                metrics = self.val_metrics
                
            acc = metrics['acc'](bin_preds, targets)
            prec = metrics['prec'](bin_preds, targets)
            rec = metrics['rec'](bin_preds, targets)
            f1 = metrics['f1'](bin_preds, targets)
            
        return acc, prec, rec, f1



    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        # Reset metrics at start of epoch
        for metric in self.train_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Update metrics
            self.train_metrics['acc'].update(outputs, targets)
            self.train_metrics['prec'].update(outputs, targets)
            self.train_metrics['rec'].update(outputs, targets)
            self.train_metrics['f1'].update(outputs, targets)

        # Compute final metrics for the epoch
        final_acc = self.train_metrics['acc'].compute().item()
        final_prec = self.train_metrics['prec'].compute().item()
        final_rec = self.train_metrics['rec'].compute().item()
        final_f1 = self.train_metrics['f1'].compute().item()

        # Store averaged metrics
        n = len(self.train_loader)
        self.metrics['train_metrics']['loss'].append(epoch_loss/n)
        self.metrics['train_metrics']['acc'].append(final_acc)
        self.metrics['train_metrics']['prec'].append(final_prec)
        self.metrics['train_metrics']['rec'].append(final_rec)
        self.metrics['train_metrics']['f1'].append(final_f1)



    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_loss = 0
        
        # Reset metrics at start of validation
        for metric in self.val_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            epoch_loss += loss.item()
            
            # Update metrics using ModuleDict
            bin_preds = (outputs > self.threshold).float()
            self.val_metrics['acc'].update(bin_preds, targets)
            self.val_metrics['prec'].update(bin_preds, targets)
            self.val_metrics['rec'].update(bin_preds, targets)
            self.val_metrics['f1'].update(bin_preds, targets)

        # Compute final metrics
        final_acc = self.val_metrics['acc'].compute().item()
        final_prec = self.val_metrics['prec'].compute().item()
        final_rec = self.val_metrics['rec'].compute().item()
        final_f1 = self.val_metrics['f1'].compute().item()

        # Store metrics with proper averaging
        n = len(self.val_loader)
        self.metrics['val_metrics']['loss'].append(epoch_loss/n)
        self.metrics['val_metrics']['acc'].append(final_acc)
        self.metrics['val_metrics']['prec'].append(final_prec)
        self.metrics['val_metrics']['rec'].append(final_rec)
        self.metrics['val_metrics']['f1'].append(final_f1)

        # Reset metrics for next validation
        for metric in self.val_metrics.values():
            metric.reset()

        return final_f1

    

    def fit(self, epochs, early_stop=5):
        best_f1 = 0
        no_improve = 0
        
        for epoch in range(epochs):
            self.train_epoch()
            val_f1 = self.validate()
            
            # Update scheduler (paper §4)
            self.scheduler.step(val_f1)
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improve += 1
                
            if no_improve >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            self._print_metrics(epoch+1)

