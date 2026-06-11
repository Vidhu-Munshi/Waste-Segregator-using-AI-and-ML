# Waste Segregator using AI and Machine Learning

## Overview

Waste Segregator using AI and Machine Learning is a deep learning-based waste classification system designed to identify common categories of waste and support efficient waste management practices. The project uses a Convolutional Neural Network (CNN) to classify waste images into predefined categories, helping improve waste segregation and recycling efforts.

The model has been trained on a custom waste image dataset and achieves approximately **80% classification accuracy** on the test dataset.

---

## Features

* Waste image classification using Deep Learning
* CNN-based image recognition model
* Classification of common waste categories
* Flask-based backend integration
* Dataset preprocessing and augmentation
* Training and prediction pipelines
* User-friendly web interface

---

## Technology Stack

* Python
* TensorFlow / Keras
* CNN (Convolutional Neural Network)
* Flask
* HTML
* CSS
* JavaScript
* Jupyter Notebook

---

## Project Structure

```text
Capstone_4thsem/
│
├── dataset/
├── modified-dataset/
├── prepared_dataset/
├── templates/
├── static/
├── models/
├── uploads/
│
├── app.py
├── predict.py
├── train_model.py
├── model.h5
├── class_names.json
└── requirements.txt
```

---

## Dataset

The model was trained on a dataset containing multiple categories of household waste materials. Images were preprocessed and organized into class-specific folders before training.

Example waste categories include:

* Plastic
* Paper
* Cardboard
* Glass
* Metal
* Other common waste materials

---

## Model Training

The CNN model was trained using TensorFlow/Keras with image preprocessing and augmentation techniques to improve generalization.

Training pipeline:

1. Data collection
2. Data preprocessing
3. Data augmentation
4. CNN model training
5. Validation and evaluation
6. Model export (`model.h5`)

---

## Results

| Metric     | Value            |
| ---------- | ---------------- |
| Model Type | CNN              |
| Accuracy   | ~80%             |
| Framework  | TensorFlow/Keras |

The model demonstrates reliable performance for basic waste classification tasks and can be further improved with a larger dataset and additional training.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Vidhu-Munshi/Waste-Segrator-using-Ai-and-Ml.git
cd Waste-Segrator-using-Ai-and-Ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## Future Improvements

* Higher classification accuracy
* Real-time camera integration
* Mobile application support
* IoT-enabled smart waste bins
* Advanced transfer learning models (EfficientNet, ResNet, MobileNet)
* Multi-object waste detection

---

## Author

**Vidhu Anand Munshi**

Engineering Student | AI & Machine Learning Enthusiast

---

## License

This project is licensed under the MIT License.
