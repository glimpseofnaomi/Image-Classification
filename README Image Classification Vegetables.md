
#  Image Classification of Vegetables using CNN

This project focuses on building a Convolutional Neural Network (CNN) to classify images of 15 different types of vegetables. The model is trained using TensorFlow and includes various data preprocessing and augmentation techniques to improve performance.

---

## Project Overview

The goal of this project is to accurately classify vegetable images into the following categories:

- Cauliflower
- Cucumber
- Papaya
- Tomato
- Broccoli
- Bean
- Carrot
- Bottle Gourd
- Bitter Gourd
- Capsicum
- Pumpkin
- Radish
- Cabbage
- Potato
- Brinjal

This classification system can be helpful in agricultural applications, automated sorting systems, or educational tools.

---

## Dataset

The dataset is sourced from Kaggle:  
🔗 [Vegetable Image Dataset by Misrak Ahmed](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

The dataset includes 15 classes of vegetables with thousands of labeled images for training and testing.

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - TensorFlow 2.15.0
  - NumPy
  - OpenCV
  - Scikit-Image
  - Pandas
  - Matplotlib, Seaborn
  - TensorFlow.js (for model deployment to web)
- **Image Preprocessing:**
  - Resize, rotate, affine transforms
  - Histogram equalization
- **Training Utilities:**
  - EarlyStopping
  - ModelCheckpoint

---

## Model Training

- CNN architecture designed to extract spatial features from vegetable images
- Data Augmentation applied to enhance generalization
- Evaluation metrics: Accuracy, Confusion Matrix, Loss Graphs
- Callback functions to monitor and optimize training

---

## Getting Started

### Installation

Ensure the following packages are installed:

```bash
pip install opencv-python scikit-image tensorflow==2.15.0 tensorflowjs
```

### Running the Notebook

1. Download or clone this repository.
2. Place the dataset inside the `dataset/` directory.
3. Open the notebook `Notebook Image Classification Vegetables.ipynb` using Jupyter.
4. Run all cells to train and evaluate the model.



## Sample Results

- Training and Validation Accuracy and Loss plotted using Matplotlib
- Model exported in `.h5` and TensorFlow.js format
- Inference on test images with prediction labels

---

## Author

**Naomi Sitanggang**  
Email: naomistg5@gmail.com  
Dicoding ID: MC006D5X1986

---

