"""
Configuration file for Lung Cancer Detection System
"""
import os

# Data paths
DATA_CONFIG = {
    'LUNA16_DATA_PATH': 'data/luna16/',
    'ANNOTATIONS_FILE': 'data/annotations.csv',
    'PROCESSED_DATA_PATH': 'processed_data/',
    'MODEL_SAVE_PATH': 'models/',
    'RESULTS_PATH': 'results/'
}

# Model configuration
MODEL_CONFIG = {
    'INPUT_SIZE': (512, 512),
    'INPUT_CHANNELS': 1,
    'NUM_CLASSES': 2,
    'DROPOUT_RATE': 0.5,
    'FPN_OUT_CHANNELS': 256,
    'ATTENTION_REDUCTION_RATIO': 16
}

# Training configuration
TRAINING_CONFIG = {
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 1e-4,
    'SCHEDULER_STEP_SIZE': 15,
    'SCHEDULER_GAMMA': 0.1,
    'NUM_WORKERS': 4,
    'PIN_MEMORY': True,
    'EARLY_STOPPING_PATIENCE': 10,
    'MIN_DELTA': 0.001
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    'TARGET_SIZE': (512, 512),
    'HU_MIN': -1000,
    'HU_MAX': 400,
    'NORMALIZATION_RANGE': (0, 255),
    'LUNG_THRESHOLD': -320,
    'MIN_OBJECT_SIZE': 1000,
    'MORPHOLOGY_DISK_SIZE': 10,
    'ROI_PADDING': 20
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'AUGMENTATION_FACTOR': 3,
    'ROTATION_RANGE': (-15, 15),
    'SCALE_RANGE': (0.9, 1.1),
    'FLIP_PROBABILITY': 0.5,
    'NOISE_FACTOR': 0.1,
    'NOISE_PROBABILITY': 0.5
}

# Focal loss configuration
LOSS_CONFIG = {
    'FOCAL_ALPHA': 1.0,
    'FOCAL_GAMMA': 2.0,
    'REDUCTION': 'mean'
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'METRICS': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
    'AVERAGE': 'weighted',
    'ZERO_DIVISION': 0
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'PAGE_TITLE': 'Lung Cancer Detection System',
    'PAGE_ICON': '🫁',
    'LAYOUT': 'wide',
    'INITIAL_SIDEBAR_STATE': 'expanded',
    'SUPPORTED_FORMATS': ['png', 'jpg', 'jpeg', 'dcm'],
    'MAX_FILE_SIZE': 50  # MB
}

# Hardware configuration
HARDWARE_CONFIG = {
    'USE_CUDA': True,
    'CUDA_VISIBLE_DEVICES': '0',
    'MIXED_PRECISION': True,
    'BENCHMARK': True
}

# Logging configuration
LOGGING_CONFIG = {
    'LOG_LEVEL': 'INFO',
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_FILE': 'logs/training.log',
    'MAX_LOG_SIZE': 10485760,  # 10MB
    'BACKUP_COUNT': 5
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    directories = [
        DATA_CONFIG['PROCESSED_DATA_PATH'],
        DATA_CONFIG['MODEL_SAVE_PATH'],
        DATA_CONFIG['RESULTS_PATH'],
        'logs/'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Validation functions
def validate_config():
    """Validate configuration parameters"""
    assert MODEL_CONFIG['INPUT_CHANNELS'] > 0, "Input channels must be positive"
    assert MODEL_CONFIG['NUM_CLASSES'] >= 2, "Number of classes must be at least 2"
    assert 0 < MODEL_CONFIG['DROPOUT_RATE'] < 1, "Dropout rate must be between 0 and 1"
    
    assert TRAINING_CONFIG['BATCH_SIZE'] > 0, "Batch size must be positive"
    assert TRAINING_CONFIG['NUM_EPOCHS'] > 0, "Number of epochs must be positive"
    assert TRAINING_CONFIG['LEARNING_RATE'] > 0, "Learning rate must be positive"
    
    print("Configuration validation passed!")

if __name__ == "__main__":
    create_directories()
    validate_config()
    print("Configuration setup completed!")