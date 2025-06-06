import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import os
import json
import cv2  # Added for image resizing
from model_architecture import LungCancerDetectionModel, FocalLoss
import warnings
warnings.filterwarnings('ignore')

class LungCancerDataset(Dataset):
    """Custom dataset for lung cancer detection"""
    
    def __init__(self, X, y, transform=None, resize_dim=224):
        self.X = X
        self.y = y
        self.transform = transform
        self.resize_dim = resize_dim
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        # Resize image to reduce memory usage
        if image.shape[0] > self.resize_dim or image.shape[1] > self.resize_dim:
            # Use simple resize for efficiency
            image = cv2.resize(image, (self.resize_dim, self.resize_dim), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.transpose(2, 0, 1)  # HWC to CHW
        
        image = torch.FloatTensor(image) / 255.0  # Normalize to [0, 1]
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ModelTrainer:
    """Training class for lung cancer detection model"""
    
    def __init__(self, model, device, save_dir='models'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Store predictions for detailed metrics
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate detailed metrics
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        return metrics, all_preds, all_targets, all_probs
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.001, 
              weight_decay=1e-4, scheduler_step_size=10, scheduler_gamma=0.1):
        """Complete training loop"""
        
        # Loss function with class weighting for imbalanced data
        criterion = FocalLoss(alpha=1, gamma=2)
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {learning_rate}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_metrics, val_preds, val_targets, val_probs = self.validate_epoch(val_loader, criterion)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1-Score: {val_metrics['f1_score']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_path = os.path.join(self.save_dir, f'best_model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_metrics['accuracy'],
                    'val_metrics': val_metrics
                }, self.best_model_path)
                print(f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%")
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
            # Early stopping check
            if epoch > 10 and val_metrics['accuracy'] < 0.5:
                print("Early stopping due to poor performance")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best model saved at: {self.best_model_path}")
        
        return self.best_model_path
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Learning curve
        plt.subplot(1, 3, 3)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        plt.fill_between(epochs, self.train_accuracies, alpha=0.3, color='blue')
        plt.fill_between(epochs, self.val_accuracies, alpha=0.3, color='red')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

class ModelEvaluator:
    """Model evaluation class"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, test_loader, save_dir='models'):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                pred = output.argmax(dim=1)
                probs = torch.softmax(output, dim=1)[:, 1]
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Print results
        print("="*50)
        print("FINAL MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Test AUC: {auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Cancer', 'Cancer'],
                   yticklabels=['No Cancer', 'Cancer'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save evaluation results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

def load_data(data_path):
    """Load processed data"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']

def main():
    # Configuration
    config = {
        'batch_size': 4,  # Reduced batch size to save memory
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'scheduler_step_size': 15,
        'scheduler_gamma': 0.1,
        'num_workers': 0,  # Reduced workers to save memory
        'pin_memory': False,  # Disabled pin_memory to save memory
        'resize_dim': 224,  # Resize images to 224x224 to save memory
        'use_smaller_model': True  # Use ResNet18 instead of ResNet50
    }
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    try:
        X_train, y_train = load_data('processed_data/train_data.pkl')
        X_val, y_val = load_data('processed_data/val_data.pkl')
        X_test, y_test = load_data('processed_data/test_data.pkl')
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
    except FileNotFoundError:
        print("Error: Processed data files not found. Please run data_preprocessing.py first.")
        return
    
    # Create datasets
    train_dataset = LungCancerDataset(X_train, y_train, resize_dim=config['resize_dim'])
    val_dataset = LungCancerDataset(X_val, y_val, resize_dim=config['resize_dim'])
    test_dataset = LungCancerDataset(X_test, y_test, resize_dim=config['resize_dim'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Create model
    print("Creating model...")
    if config['use_smaller_model']:
        # Import the smaller model variant
        from model_architecture import LungCancerDetectionModelSmall
        model = LungCancerDetectionModelSmall(num_classes=2, input_channels=1)
        print("Using smaller ResNet18-based model to reduce memory usage")
    else:
        model = LungCancerDetectionModel(num_classes=2, input_channels=1)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ModelTrainer(model, device)
    
    # Train model
    print("Starting training...")
    best_model_path = trainer.train(
        train_loader, 
        val_loader, 
        config['num_epochs'],
        config['learning_rate'],
        config['weight_decay'],
        config['scheduler_step_size'],
        config['scheduler_gamma']
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_loader)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results
    }, 'models/final_model.pth')
    
    print("Training and evaluation completed!")
    print(f"Final test accuracy: {results['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()