from datetime import datetime
from pathlib import Path


LOG_DIR = Path('logs/fit') / datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_FILE_PATH = Path('model.h5')

# model architecture parameters
INPUT_SHAPE = (32, 32, 3)
CONV_PARAMS = {
    'conv1_filters': 64, 
    'conv1_kernel_size': (3, 3),
    'conv2_filters': 64,
    'conv2_kernel_size': (3, 3),
    'maxpool_size': (2, 2),
    'dropout_rate': 0.25}
DENSE_UNITS = 512
DROPOUT_RATE = 0.5
OUTPUT_UNITS = 10

# fitting parameters
EPOCHS = 20
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
