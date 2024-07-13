SVM Prediction Project:
This project uses a Support Vector Machine (SVM) to classify images of cats and dogs based on their features extracted using Histogram of Oriented Gradients (HOG). The project leverages Python libraries such as OpenCV, NumPy, Pandas, scikit-image, and scikit-learn.

Project Structure:
The project includes scripts for loading and processing images, training an SVM model, evaluating the model, and generating predictions for submission.

Libraries Used:
OpenCV
NumPy
Pandas
scikit-image
scikit-learn
joblib
Getting Started
Prerequisites

Ensure you have the following libraries installed:
pip install opencv-python numpy pandas scikit-image scikit-learn joblib

Project Files:
train: Directory containing training images of cats and dogs.
test1: Directory containing test images of cats and dogs.
sampleSubmission.csv: CSV file to store the predictions.

How It Works:
Data Loading and Preprocessing: The images are loaded from the specified folders, resized, and HOG features are extracted.
Model Training: An SVM model with a linear kernel is trained using the extracted features.
Model Evaluation: The model is evaluated on the test dataset, and performance metrics are printed.
Prediction and Submission: Predictions are generated for the test images, and a submission file is created in CSV format.

Acknowledgements:
OpenCV: An open-source computer vision and machine learning software library.
NumPy: A fundamental package for scientific computing with Python.
Pandas: An open-source data analysis and manipulation tool.
scikit-image: A collection of algorithms for image processing in Python.
scikit-learn: A machine learning library for Python.
