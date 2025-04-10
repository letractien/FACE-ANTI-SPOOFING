# Face Liveness Detection using YOLOv8

This project implements a face liveness detection system using YOLOv8. It can distinguish between real faces and fake/spoofed faces in real-time using a webcam.

## Project Structure

```
FACE-ANTI-SPOOFING/           # Project root directory
├── Dataset/                  # Data directory
│   ├── Data/                # Original available data
│   ├── DataStandard/        # Standardized data from Data
│   ├── DataCollect/         # Labeled data from DataStandard
│   └── SplitData/           # Split data for fine-tuning
│       ├── train/           # Training data
│       ├── val/             # Validation data
│       └── test/            # Test data
│
├── models/                   # Directory for trained models
│   └── best.pt              # Best performing model after fine-tuning
│
├── runs/                     # Training results directory
│   └── detect/              # Detection results
│       └── train/           # Training results
│           ├── weights/     # Weight files
│           └── results.csv  # Metrics results
│
├── data_collection_1.py     # Data collection script method 1 (from Video)
├── data_collection_2.py     # Data collection script method 2 (from available data)
├── chuan_hoa_anh.py        # Image standardization script
├── split_data.py           # Train/val/test data split script
├── train.py                # Model training script
├── liveness.py             # Main face liveness detection script
├── check_cam.py            # Camera check script
└── yolov8n.pt             # Pre-trained YOLOv8 model
```

## Dataset Structure

The Dataset directory follows a pipeline of data processing:

```
Dataset/
├── Data/                   # Original available data
│   ├── normal/             # Original images
│   └── spoof/              # Original labels
│
├── DataStandard/           # Standardized data
│   ├── normal/             # Standardized images
│   └── spoof/              # Standardized labels
│
├── DataCollect/            # Labeled data
│   ├── images/             # Labeled images
│   └── labels/             # Label files
│
└── SplitData/              # Split dataset for training
    ├── train/              # Training set (70%)
    │   ├── images/         # Training images
    │   └── labels/         # Training labels
    │
    ├── val/                # Validation set (20%)
    │   ├── images/         # Validation images
    │   └── labels/         # Validation labels
    │
    └── test/               # Test set (10%)
        ├── images/         # Test images
        └── labels/         # Test labels
```

### Data Processing Pipeline

1. **Data/**
   - Contains the original available dataset
   - Raw images and their corresponding labels
   - Starting point of the data processing pipeline

2. **DataStandard/**
   - Contains standardized data from Data
   - Images are normalized and resized
   - Ensures consistent image dimensions and quality

3. **DataCollect/**
   - Contains labeled data from DataStandard
   - Uses FaceDetector for automatic face labeling
   - Each image has a corresponding label file

4. **SplitData/**
   - Final dataset split for model training
   - Divided into:
     - Training set (70%): Used for model training
     - Validation set (20%): Used for model validation
     - Test set (10%): Used for final model evaluation

### Data Format

- Images: JPG/PNG format
- Labels: YOLO format (.txt files)
  - Each line represents one object in the image
  - Format: `class_id x_center y_center width height`
  - Coordinates are normalized (0-1)
  - Class IDs:
    - 0: fake face
    - 1: real face

## Hardware Requirements

- Python 3.8 or higher
- Webcam
- CUDA-supported GPU (recommended for better performance)

## Installation Guide

### Step 1: Install Python
1. Visit https://www.python.org/downloads/
2. Download and install Python version 3.8 or higher
3. During installation, check "Add Python to PATH"

### Step 2: Create Virtual Environment
1. Open PowerShell or Command Prompt
2. Navigate to the project directory
3. Create virtual environment:
```bash
python -m venv venv
```
4. Activate virtual environment:
```bash
.\venv\Scripts\activate
```

### Step 3: Install Required Libraries
1. Ensure virtual environment is activated
2. Install required libraries:
```bash
pip install -r requirements.txt
```

### Step 4: Camera Check
1. Test webcam:
```bash
python check_cam.py
```
2. If working properly, you will see the IDs of available webcams

### Step 5: Data Normalization
```bash
python chuan_hoa_anh.py
```
- This step normalizes the data to ensure proper resizing ratios

### Step 6: Automatic Labeling
```bash
python data_collection_2.py
```
- Uses the pre-trained FaceDetector model for quick face labeling

### Step 7: Split Data for Training and Testing
```bash
python split_data.py
```
- Splits data into:
  - Training set: 70%
  - Validation set: 20%
  - Test set: 10%

### Step 8: Fine-tune Model
1. In the SplitData directory, copy data.yml to dataOffline.yml
2. Open dataOffline.yml and adjust the path variable to point to the current SplitData directory location:
```yaml
path: C:\Users\wwhac\Desktop\Cake\yolo\Dataset\SplitData
train: train/images
val: val/images
test: test/images

nc: 2
names: ['fake', 'real']
```
3. Start fine-tuning:
```bash
python train.py
```
- Creates a 'runs' directory to store training results
- Best model checkpoints are saved in runs/detect/train/weights/
- Each training run creates a new directory (train, train2, train3, etc.)

### Step 9: Run the Model
```bash
python liveness.py
```

## Usage

After completing the installation and training steps, run `liveness.py` to start the face liveness detection system. The system will use your webcam to detect faces and determine if they are real or fake in real-time.

## Important Notes

- Ensure you have a trained model in the `runs/detect/train/weights/` directory
- The system requires a webcam for real-time detection
- For best results, ensure good lighting conditions during data collection and detection
- Each training run creates a new directory in the runs/detect/ folder

## Troubleshooting

If you encounter issues:
1. Verify that your webcam is properly connected and recognized
2. Confirm all required libraries are correctly installed
3. Ensure model files are in the correct directories
4. For CUDA-related errors, check NVIDIA driver and CUDA Toolkit installation 