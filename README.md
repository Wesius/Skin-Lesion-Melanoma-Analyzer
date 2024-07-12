# Skin Lesion Classification Project

## Project Overview

This project implements a deep learning-based system for classifying skin lesions as either melanoma or non-melanoma. It utilizes state-of-the-art computer vision techniques and a convolutional neural network (CNN) architecture to analyze images of skin lesions and provide diagnoses.

### Key Features

- Data preprocessing and augmentation for balanced dataset creation
- Custom CNN model based on EfficientNetV2-S architecture
- Training pipeline with mixed precision and gradient accumulation
- Model evaluation and performance metrics
- Web-based interface for real-time image classification

### Technologies Used

- Python 3.x
- PyTorch
- torchvision
- Flask
- Albumentations
- scikit-learn
- PIL (Python Imaging Library)
- NumPy

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [Code Structure](#code-structure)
5. [Detailed Functionality Explanation](#detailed-functionality-explanation)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Performance Considerations](#performance-considerations)
9. [Contributing](#contributing)
10. [Changelog](#changelog)
11. [License](#license)
12. [Contact Information](#contact-information)

## Installation

Follow these steps to set up the project:

1. Clone the repository:
   ```
   git clone https://github.com/Wesius/skin-lesion-classification.git
   cd skin-lesion-classification
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle:
   - Go to https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification
   - Download and extract the ZIP file
   - Move the `images` folder and `GroundTruth.csv` file to the project root directory

5. Install CUDA Toolkit (if you have a CUDA-enabled GPU):
   - Go to https://developer.nvidia.com/cuda-toolkit
   - Download and install the appropriate version for your system

6. Go to [Usage](#usage) for instructions on data processing, model training, model analysis and the web server. 


## Usage

### Data Preparation

1. Run the data preparation script:
   ```
   python prepdata.py
   ```
   This script will load the data, split it into train/validation/test sets, balance the dataset, and apply augmentations.

### Model Training

1. Train the model using:
   ```
   python model.py
   ```
   This script will train the CNN model on the prepared dataset and save the best model weights.

### Analysis

1. Run the analysis script:
   ```
   python analysis.py
   ```
   This script will perform additional analysis on the dataset and model performance, generating visualizations and statistics.

### Web Application

1. Start the Flask web application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload an image of a skin lesion to get a classification result

## Configuration

The project uses several configuration parameters that can be adjusted in the respective Python files:

### `prepdata.py`

- `csv_path`: Path to the ground truth CSV file
- `image_dir`: Directory containing the original images
- `output_base_dir`: Directory for the prepared dataset

### `model.py`

- `batch_size`: Number of images per batch during training
- `num_epochs`: Number of training epochs
- `patience`: Number of epochs to wait before early stopping
- `accumulation_steps`: Number of steps for gradient accumulation

### `app.py`

- `debug`: Set to `True` for development, `False` for production

To modify these parameters, open the respective files and adjust the values as needed.

## Code Structure

The project consists of the following main files:

- `prepdata.py`: Data preparation and augmentation
- `model.py`: Model definition, training, and evaluation
- `app.py`: Flask web application for serving predictions
- `analysis.py`: Additional analysis and visualization tools
- `templates/upload.html`: HTML template for the web interface

### `prepdata.py`

This script handles data loading, splitting, augmentation, and preparation. Key functions include:

- `load_data()`: Loads image paths and labels from the CSV file
- `split_data()`: Splits the dataset into train, validation, and test sets
- `augment_melanoma_images()`: Applies data augmentation to balance the dataset
- `balance_dataset()`: Ensures class balance in the dataset
- `prepare_dataset()`: Copies and renames images to the prepared dataset directory

### `model.py`

This file defines the model architecture, training loop, and evaluation functions:

- `SkinLesionDataset`: Custom PyTorch dataset class for skin lesion images
- `SkinLesionModel`: CNN model based on EfficientNetV2-S
- `train_model()`: Training loop with mixed precision and gradient accumulation
- `evaluate_model()`: Model evaluation on the test set

### `app.py`

The Flask application for serving the model:

- Loads the trained model
- Defines routes for the web interface
- Handles image upload and prediction

### `analysis.py`

Additional tools for dataset analysis and visualization:

- Generates distribution plots of the dataset
- Creates confusion matrices
- Performs error analysis on misclassified images

## Detailed Functionality Explanation

### Data Preparation and Augmentation

The data preparation process involves several steps:

1. **Loading Data**: The `load_data()` function reads the CSV file containing image names and labels, creating a list of image paths and corresponding labels.

2. **Splitting Data**: The dataset is randomly shuffled and split into train (70%), validation (20%), and test (10%) sets using the `split_data()` function.

3. **Balancing Dataset**: To address class imbalance, the `balance_dataset()` function is used. If there are fewer melanoma images than non-melanoma:
   - Melanoma images are augmented using the `augment_melanoma_images()` function
   - Augmentation techniques include rotation, flipping, distortion, and color jittering

4. **Preparing Final Dataset**: The `prepare_dataset()` function copies images to their respective directories (train/val/test) with appropriate prefixes ('mel_' for melanoma, 'nomel_' for non-melanoma).

### Model Architecture and Training

The model uses the EfficientNetV2-S architecture, pretrained on ImageNet and fine-tuned for binary classification:

1. **Model Definition**: The `SkinLesionModel` class defines the model structure, replacing the final classification layer with a new one for binary classification.

2. **Training Loop**: The `train_model()` function implements the training process:
   - Uses mixed precision training for efficiency
   - Implements gradient accumulation to simulate larger batch sizes
   - Employs early stopping to prevent overfitting
   - Uses a learning rate scheduler to adjust the learning rate during training

3. **Evaluation**: The `evaluate_model()` function assesses the model's performance on the test set, providing accuracy and a detailed classification report.

### Analysis

The `analysis.py` script provides additional insights into the dataset and model performance:

1. **Data Distribution**: Generates plots showing the distribution of classes in the dataset.

2. **Confusion Matrix**: Creates a confusion matrix to visualize the model's performance across different classes.

3. **Error Analysis**: Identifies and displays examples of misclassified images for further investigation.

### Web Application

The Flask application (`app.py`) serves as an interface for real-time predictions:

1. **Model Loading**: The trained model is loaded at application startup.

2. **Image Processing**: Uploaded images are preprocessed using the same transformations as during training.

3. **Prediction**: The model generates a prediction for the uploaded image, classifying it as either melanoma or non-melanoma.

4. **Result Display**: The classification result is sent back to the web interface and displayed to the user.

## Examples

### Preparing the Dataset

```python
python prepdata.py
```

This will create a balanced dataset in the `prepared_dataset` directory with the following structure:

```
prepared_dataset/
├── train/
│   ├── mel_image1.jpg
│   ├── nomel_image2.jpg
│   └── ...
├── val/
│   ├── mel_image3.jpg
│   ├── nomel_image4.jpg
│   └── ...
└── test/
    ├── mel_image5.jpg
    ├── nomel_image6.jpg
    └── ...
```

### Training the Model

```python
python model.py
```

This will train the model and output training progress:

```
Epoch [1/10] - Train Loss: 0.6932, Train Accuracy: 52.14%, Val Loss: 0.6845, Val Accuracy: 55.32%
Epoch [2/10] - Train Loss: 0.6523, Train Accuracy: 61.87%, Val Loss: 0.6421, Val Accuracy: 63.75%
...
```

### Running Analysis

```python
python analysis.py
```

This will generate various plots and statistics, such as:

```
Class Distribution:
Melanoma: 25.3%
Non-Melanoma: 74.7%

Model Performance:
Accuracy: 87.6%
Precision: 0.83
Recall: 0.79
F1-Score: 0.81

Confusion Matrix:
[[892  58]
 [ 76 274]]
```

### Using the Web Application

1. Start the Flask app:
   ```
   python app.py
   ```

2. Open a web browser and go to `http://localhost:5000`

3. Upload an image of a skin lesion

4. Receive a classification result:
   ```
   Classification Result: Melanoma
   ```

## Troubleshooting

- **CUDA out of memory error**: Reduce the batch size in `model.py` or increase gradient accumulation steps
- **Image not found error**: Ensure all images referenced in the CSV file are present in the `images` directory
- **Model not found error**: Make sure you've trained the model and generated the `best_model.pth` file before running the web application

## Performance Considerations

- The use of mixed precision training and gradient accumulation helps in training larger models efficiently
- EfficientNetV2-S provides a good balance between accuracy and computational requirements
- For inference on CPU, consider using a smaller model or quantization techniques

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

## Changelog

- v1.0.0 (2023-06-25)
  - Initial release with 7 different classifications
  - Model accuracy around 45%
- v1.1.0 (2023-07-03)
  - Changed from 7 classes to 2 (melanoma/non-melanoma)
  - Significantly improved model accuracy

## License

This project is licensed under the MIT License:

Copyright (c) 2023 Wes Griffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact Information

For questions or support, please contact:

- Project Maintainer: Wes Griffin (wesgriffin32@gmail.com)
- Project Repository: [https://github.com/Wesius/Skin-Lesion-Melanoma-Analyzer](https://github.com/Wesius/Skin-Lesion-Melanoma-Analyzer)
