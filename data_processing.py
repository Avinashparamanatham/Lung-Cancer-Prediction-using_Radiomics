import os
import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

class LungSegmentation:
    """Lung segmentation and preprocessing for LUNA16 dataset"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.target_size = (512, 512)
        
    def load_scan(self, path):
        """Load CT scan from file"""
        try:
            scan = sitk.ReadImage(path)
            scan_array = sitk.GetArrayFromImage(scan)
            return scan_array
        except Exception as e:
            print(f"Error loading scan {path}: {e}")
            return None
    
    def normalize_hu(self, image):
        """Normalize HU values"""
        # Clip values between -1000 and 400 HU
        image = np.clip(image, -1000, 400)
        # Normalize to 0-255
        image = (image + 1000) / 1400 * 255
        return image.astype(np.uint8)
    
    def segment_lungs(self, image, threshold=-320):
        """Segment lungs from CT scan"""
        # Convert to binary image
        binary = image < threshold
        
        # Remove small objects
        cleared = morphology.remove_small_objects(binary, min_size=1000)
        
        # Label connected components
        label_image = measure.label(cleared)
        
        # Find the two largest components (lungs)
        regions = measure.regionprops(label_image)
        regions.sort(key=lambda x: x.area, reverse=True)
        
        # Create lung mask
        lung_mask = np.zeros_like(image, dtype=bool)
        for region in regions[:2]:  # Take two largest regions
            lung_mask[label_image == region.label] = True
        
        # Apply morphological operations
        lung_mask = morphology.binary_closing(lung_mask, morphology.disk(10))
        lung_mask = binary_fill_holes(lung_mask)
        
        return lung_mask
    
    def extract_lung_roi(self, image, mask):
        """Extract lung region of interest"""
        # Apply mask
        masked_image = image.copy()
        masked_image[~mask] = 0
        
        # Find bounding box
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return image
        
        min_row, max_row = coords[0].min(), coords[0].max()
        min_col, max_col = coords[1].min(), coords[1].max()
        
        # Extract ROI with padding
        padding = 20
        min_row = max(0, min_row - padding)
        max_row = min(image.shape[0], max_row + padding)
        min_col = max(0, min_col - padding)
        max_col = min(image.shape[1], max_col + padding)
        
        roi = masked_image[min_row:max_row, min_col:max_col]
        return roi
    
    def resize_image(self, image, target_size=(512, 512)):
        """Resize image to target size"""
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def preprocess_slice(self, slice_image):
        """Complete preprocessing pipeline for a single slice"""
        # Normalize HU values
        normalized = self.normalize_hu(slice_image)
        
        # Segment lungs
        lung_mask = self.segment_lungs(normalized)
        
        # Extract lung ROI
        lung_roi = self.extract_lung_roi(normalized, lung_mask)
        
        # Resize to target size
        resized = self.resize_image(lung_roi, self.target_size)
        
        return resized
    
    def process_dataset(self, annotations_file):
        """Process entire LUNA16 dataset"""
        # Load annotations
        annotations = pd.read_csv(annotations_file)
        
        processed_data = []
        labels = []
        
        print("Processing LUNA16 dataset...")
        
        for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
            series_uid = row['seriesuid']
            coord_x = row['coordX']
            coord_y = row['coordY']
            coord_z = row['coordZ']
            diameter = row['diameter_mm']
            
            # Determine label (nodule vs non-nodule)
            label = 1 if diameter > 3 else 0  # Nodules > 3mm considered positive
            
            # Load corresponding CT scan
            scan_path = os.path.join(self.data_path, f"{series_uid}.mhd")
            
            if os.path.exists(scan_path):
                scan = self.load_scan(scan_path)
                if scan is not None:
                    # Extract relevant slices around the nodule
                    slice_idx = int(coord_z)
                    
                    # Process multiple slices around the nodule
                    for offset in [-2, -1, 0, 1, 2]:
                        current_slice = slice_idx + offset
                        if 0 <= current_slice < scan.shape[0]:
                            slice_image = scan[current_slice]
                            processed_slice = self.preprocess_slice(slice_image)
                            
                            processed_data.append(processed_slice)
                            labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(processed_data)
        y = np.array(labels)
        
        # Add channel dimension for CNN
        X = np.expand_dims(X, axis=-1)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Positive samples: {np.sum(y)}")
        print(f"Negative samples: {len(y) - np.sum(y)}")
        
        return X, y
    
    def save_processed_data(self, X, y, filename):
        """Save processed data"""
        os.makedirs(self.output_path, exist_ok=True)
        
        data_dict = {
            'X': X,
            'y': y
        }
        
        with open(os.path.join(self.output_path, filename), 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Data saved to {os.path.join(self.output_path, filename)}")

class DataAugmentation:
    """Data augmentation for lung CT images"""
    
    def __init__(self):
        pass
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))
    
    def flip_image(self, image, flip_code):
        """Flip image horizontally or vertically"""
        return cv2.flip(image, flip_code)
    
    def scale_image(self, image, scale_factor):
        """Scale image by given factor"""
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        scaled = cv2.resize(image, (new_width, new_height))
        
        # Crop or pad to original size
        if scale_factor > 1:
            # Crop center
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            return scaled[start_y:start_y+height, start_x:start_x+width]
        else:
            # Pad with zeros
            pad_y = (height - new_height) // 2
            pad_x = (width - new_width) // 2
            return cv2.copyMakeBorder(scaled, pad_y, height-new_height-pad_y,
                                    pad_x, width-new_width-pad_x, cv2.BORDER_CONSTANT)
    
    def add_noise(self, image, noise_factor=0.1):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_factor * 255, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def augment_dataset(self, X, y, augmentation_factor=2):
        """Augment dataset with various transformations"""
        augmented_X = []
        augmented_y = []
        
        # Get the expected shape from the original data
        expected_shape = X.shape[1:]  # (512, 512, 1) or similar
        
        # Original data
        augmented_X.extend(X)
        augmented_y.extend(y)
        
        print("Augmenting dataset...")
        
        for i in tqdm(range(len(X))):
            image = X[i]
            label = y[i]
            
            for _ in range(augmentation_factor):
                # Random augmentation
                aug_image = image.copy()
                
                # Random rotation
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    aug_image = self.rotate_image(aug_image, angle)
                
                # Random flip
                if np.random.random() > 0.5:
                    flip_code = np.random.choice([0, 1])  # 0: vertical, 1: horizontal
                    aug_image = self.flip_image(aug_image, flip_code)
                
                # Random scaling
                if np.random.random() > 0.5:
                    scale = np.random.uniform(0.9, 1.1)
                    aug_image = self.scale_image(aug_image, scale)
                
                # Random noise
                if np.random.random() > 0.5:
                    aug_image = self.add_noise(aug_image)
                
                # Ensure the augmented image has the expected shape
                if aug_image.shape != expected_shape:
                    # If the channel dimension is missing, add it back
                    if len(aug_image.shape) == 2:
                        aug_image = np.expand_dims(aug_image, axis=-1)
                    # Resize if dimensions don't match
                    elif aug_image.shape[:2] != expected_shape[:2]:
                        aug_image = cv2.resize(aug_image, expected_shape[:2][::-1])
                        # Add channel dimension if needed
                        if len(aug_image.shape) == 2:
                            aug_image = np.expand_dims(aug_image, axis=-1)
                
                augmented_X.append(aug_image)
                augmented_y.append(label)
        
        # Convert to numpy arrays with explicit dtype
        augmented_X_array = np.array(augmented_X, dtype=np.float32)
        augmented_y_array = np.array(augmented_y, dtype=np.int32)
        
        return augmented_X_array, augmented_y_array

def main():
    # Configuration
    data_path = "seg-lungs-LUNA16/seg-lungs-LUNA16"  # Path to LUNA16 dataset
    output_path = "processed_data"
    annotations_file = "annotations.csv"  # Annotations file in current directory
    
    # Initialize preprocessor
    preprocessor = LungSegmentation(data_path, output_path)
    
    # Process dataset
    X, y = preprocessor.process_dataset(annotations_file)
    
    # Initialize augmentation
    augmenter = DataAugmentation()
    
    # Augment dataset
    X_aug, y_aug = augmenter.augment_dataset(X, y, augmentation_factor=3)
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_aug, y_aug, test_size=0.3, random_state=42, stratify=y_aug
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Save processed datasets
    preprocessor.save_processed_data(X_train, y_train, 'train_data.pkl')
    preprocessor.save_processed_data(X_val, y_val, 'val_data.pkl')
    preprocessor.save_processed_data(X_test, y_test, 'test_data.pkl')
    
    print("Data preprocessing completed!")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

if __name__ == "__main__":
    main()