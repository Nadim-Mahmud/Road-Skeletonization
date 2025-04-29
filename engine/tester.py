import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

class Tester:
    def __init__(self, model, test_loader, device, threshold=0.5):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        
        # Metrics
        self.metrics = {
            'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0
        }
        
    def test(self):
        self.model.eval()
        acc_meter, prec_meter, rec_meter, f1_meter = 0, 0, 0, 0
        
        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                bin_preds = (outputs > self.threshold).float()
                
                # Update metrics
                acc_meter += Accuracy(task='binary')(bin_preds, targets).item()
                prec_meter += Precision(task='binary')(bin_preds, targets).item()
                rec_meter += Recall(task='binary')(bin_preds, targets).item()
                f1_meter += F1Score(task='binary')(bin_preds, targets).item()
                
        n = len(self.test_loader)
        self.metrics['acc'] = acc_meter/n
        self.metrics['prec'] = prec_meter/n
        self.metrics['rec'] = rec_meter/n
        self.metrics['f1'] = f1_meter/n
        
        return self.metrics
    
    def visualize_samples(self, num_samples=3):
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        self.model.eval()
        
        with torch.no_grad():
            for i, (inputs, targets, _) in enumerate(self.test_loader):
                if i >= num_samples:
                    break
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = (outputs > self.threshold).float().cpu()
                
                # Denormalize
                img = inputs[0].cpu().numpy().squeeze()
                img = (img * 0.5 + 0.5) * 255
                
                axes[i,0].imshow(img, cmap='gray')
                axes[i,0].set_title('Input')
                axes[i,1].imshow(targets[0].squeeze(), cmap='gray')
                axes[i,1].set_title('Ground Truth')
                axes[i,2].imshow(preds[0].squeeze(), cmap='gray')
                axes[i,2].set_title('Prediction')
                
        plt.tight_layout()
        plt.savefig('test_samples.png')
        plt.close()
