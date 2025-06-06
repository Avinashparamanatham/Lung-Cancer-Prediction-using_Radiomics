"""
Utility functions for lung cancer detection system
"""
import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import logging
from typing import Dict, List, Tuple, Any
import cv2
from PIL import Image
import SimpleITK as sitk

# Setup logging
def setup_logging(log_file='logs/system.log', log_level=logging.INFO):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class MetricsCalculator:
    """Calculate and store various evaluation metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, matthews_corrcoef
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = 0.0
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        self.metrics = metrics
        return metrics
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report"""
        target_names = ['No Cancer', 'Cancer']
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None) -> None:
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Cancer', 'Cancer'],
                   yticklabels=['No Cancer', 'Cancer'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_prediction_distribution(predictions: np.ndarray, true_labels: np.ndarray,
                                   save_path: str = None) -> None:
        """Plot prediction distribution"""
        plt.figure(figsize=(12, 4))
        
        # Prediction histogram for each class
        plt.subplot(1, 2, 1)
        no_cancer_preds = predictions[true_labels == 0]
        cancer_preds = predictions[true_labels == 1]
        
        plt.hist(no_cancer_preds, bins=20, alpha=0.7, label='No Cancer', color='green')
        plt.hist(cancer_preds, bins=20, alpha=0.7, label='Cancer', color='red')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution by True Class')
        plt.legend()
        plt.grid(True)
        
        # Box plot
        plt.subplot(1, 2, 2)
        data = [no_cancer_preds, cancer_preds]
        labels = ['No Cancer', 'Cancer']
        plt.boxplot(data, labels=labels)
        plt.ylabel('Prediction Probability')
        plt.title('Prediction Distribution Box Plot')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_feature_map_visualization(feature_maps: torch.Tensor, 
                                       save_path: str = None) -> None:
        """Visualize feature maps"""
        if len(feature_maps.shape) == 4:
            # Take first sample and first few channels
            feature_maps = feature_maps[0, :16]  # First 16 channels
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(16, feature_maps.shape[0])):
            axes[i].imshow(feature_maps[i].cpu().detach().numpy(), cmap='viridis')
            axes[i].set_title(f'Feature Map {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(feature_maps.shape[0], 16):
            axes[i].axis('off')
        
        plt.suptitle('Feature Map Visualization')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return self.early_stop

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate class activation map"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Normalize
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam
    
    def visualize_cam(self, input_image: np.ndarray, cam: np.ndarray,
                     save_path: str = None) -> None:
        """Visualize CAM overlay"""
        # Resize CAM to input size
        cam_resized = cv2.resize(cam, input_image.shape[::-1])
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Overlay
        if len(input_image.shape) == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        
        overlay = heatmap * 0.4 + input_image * 0.6
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Class Activation Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title('CAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Additional utility functions
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")

def get_device() -> torch.device:
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device

def memory_usage() -> Dict[str, float]:
    """Get memory usage information"""
    import psutil
    
    # System memory
    mem = psutil.virtual_memory()
    system_memory = {
        'total': mem.total / (1024**3),  # GB
        'available': mem.available / (1024**3),  # GB
        'percent': mem.percent
    }
    
    # GPU memory (if available)
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
            'cached': torch.cuda.memory_reserved() / (1024**3),  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)  # GB
        }
    
    return {
        'system': system_memory,
        'gpu': gpu_memory
    }

def create_experiment_directory(base_dir: str = 'experiments') -> str:
    """Create experiment directory with timestamp"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'logs', 'plots', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir

def export_model_to_onnx(model: torch.nn.Module, input_shape: Tuple[int, ...],
                        save_path: str) -> None:
    """Export PyTorch model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {save_path}")

if __name__ == "__main__":
    # Test utility functions
    logger.info("Testing utility functions...")
    
    # Test device detection
    device = get_device()
    
    # Test memory usage
    memory_info = memory_usage()
    logger.info(f"Memory usage: {memory_info}")
    
    # Test experiment directory creation
    exp_dir = create_experiment_directory()
    
    logger.info("Utility functions test completed!")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,save_path: str = None) -> None:
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class DataUtils:
    """Utility functions for data handling"""
    
    @staticmethod
    def save_data(data: Any, filepath: str) -> None:
        """Save data to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_data(filepath: str) -> Any:
        """Load data from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Data loaded from {filepath}")
        return data
    
    @staticmethod
    def validate_data_split(X_train: np.ndarray, X_val: np.ndarray, 
                           X_test: np.ndarray, y_train: np.ndarray,
                           y_val: np.ndarray, y_test: np.ndarray) -> None:
        """Validate data split"""
        assert len(X_train) == len(y_train), "Training data size mismatch"
        assert len(X_val) == len(y_val), "Validation data size mismatch"
        assert len(X_test) == len(y_test), "Test data size mismatch"
        
        total_samples = len(X_train) + len(X_val) + len(X_test)
        train_ratio = len(X_train) / total_samples
        val_ratio = len(X_val) / total_samples
        test_ratio = len(X_test) / total_samples
        
        logger.info(f"Data split ratios - Train: {train_ratio:.2f}, "
                   f"Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
        
        # Check class distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        logger.info(f"Training class distribution: {dict(zip(unique_train, counts_train))}")
        logger.info(f"Validation class distribution: {dict(zip(unique_val, counts_val))}")
        logger.info(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

class ImageUtils:
    """Utility functions for image processing"""
    
    @staticmethod
    def load_dicom_image(filepath: str) -> np.ndarray:
        """Load DICOM image"""
        try:
            image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(image)
            return array
        except Exception as e:
            logger.error(f"Error loading DICOM file {filepath}: {e}")
            return None
    
    @staticmethod
    def normalize_image(image: np.ndarray, min_val: float = -1000, 
                       max_val: float = 400) -> np.ndarray:
        """Normalize image values"""
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        return image.astype(np.float32)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size"""
        if len(image.shape) == 2:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        elif len(image.shape) == 3:
            resized_slices = []
            for i in range(image.shape[0]):
                resized_slice = cv2.resize(image[i], target_size, interpolation=cv2.INTER_AREA)
                resized_slices.append(resized_slice)
            return np.array(resized_slices)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    
    @staticmethod
    def apply_window_level(image: np.ndarray, window: float, level: float) -> np.ndarray:
        """Apply window/level adjustment for CT images"""
        min_val = level - window / 2
        max_val = level + window / 2
        
        windowed = np.clip(image, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val) * 255
        
        return windowed.astype(np.uint8)
    
    @staticmethod
    def create_image_montage(images: List[np.ndarray], grid_size: Tuple[int, int],
                           titles: List[str] = None, save_path: str = None) -> None:
        """Create image montage"""
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        
        if rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(images):
                    axes[i][j].imshow(images[idx], cmap='gray')
                    if titles and idx < len(titles):
                        axes[i][j].set_title(titles[idx])
                    axes[i][j].axis('off')
                else:
                    axes[i][j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ModelUtils:
    """Utility functions for model handling"""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    @staticmethod
    def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metrics: Dict[str, float],
                       filepath: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    @staticmethod
    def load_checkpoint(filepath: str, model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
        """Load model checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    @staticmethod
    def get_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]) -> str:
        """Get model summary"""
        from torchsummary import summary
        
        try:
            summary_str = summary(model, input_size, verbose=0)
            return str(summary_str)
        except:
            total_params, trainable_params = ModelUtils.count_parameters(model)
            return f"Total Parameters: {total_params:,}\nTrainable Parameters: {trainable_params:,}"

class VisualizationUtils:
    """Utility functions for visualization"""
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], val_losses: List[float],
                           train_accs: List[float], val_accs: List[float],
                           save_path: str = None) -> None:
        """Plot training curves"""
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()