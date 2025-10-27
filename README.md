# Face Mask Detection (CNN)

## Overview
This project implements a **Convolutional Neural Network (CNN)** to detect whether a person in an image is wearing a face mask or not. The model is trained using image data and achieves reliable accuracy for real-world applications such as public monitoring and healthcare safety systems.

## Dataset
The dataset used in this project is the **Face Mask Detection Dataset** from **Kaggle**.  
It contains two categories of images:
- With Mask
- Without Mask

Dataset Link: [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets)

## Tools & Libraries
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Model Architecture
The CNN model consists of:
- Convolutional layers for feature extraction  
- MaxPooling layers for dimensionality reduction  
- Dense layers for classification  
- Dropout layers to prevent overfitting

Optimizer: Adam  
Loss Function: Binary Crossentropy  
Evaluation Metric: Accuracy

## Results
The trained model demonstrates high accuracy on both training and validation datasets.  
It effectively distinguishes between masked and unmasked faces.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script:
   ```bash
   python face_mask_detection.py
   ```

## Future Enhancements
- Integrate real-time webcam detection  
- Deploy the model as a web app or mobile app  
- Improve accuracy using transfer learning (e.g., MobileNetV2)

## Author
**Elavarasi Chinnadurai**  
Face Mask Detection using CNN | Deep Learning Project

---
© 2025 Elavarasi Chinnadurai. All rights reserved.
