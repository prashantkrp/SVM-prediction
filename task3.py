import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import logging
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(file_info):
    """
    Process a single image: read, resize, and extract HOG features.
    """
    filename, label, folder = file_info
    img_path = os.path.join(folder, filename)

    try:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logging.warning(f"Unable to read image {img_path}. Skipping.")
            return None, None

        image = cv2.resize(image, (128, 128))  # Resize to consistent size

        # Extract HOG features
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        return features, label

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return None, None

def load_images_and_extract_features(folder, limit=None, batch_size=100):
    """
    Load images from a specified folder, extract HOG features, and return features, labels, and filenames.
    """
    logging.info(f"Loading images from folder: {folder}")

    file_infos = []
    count = 0

    for filename in os.listdir(folder):
        if limit and count >= limit:
            break

        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            if filename.startswith('cat'):
                label = 0  # Cat
            elif filename.startswith('dog'):
                label = 1  # Dog
            else:
                continue  # Skip if neither cat nor dog

            file_infos.append((filename, label, folder))
            count += 1

    if not file_infos:
        logging.error(f"No JPG or JPEG images found in folder: {folder}")
        return np.array([]), np.array([]), []

    features = []
    labels = []
    filenames = []

    for i in range(0, len(file_infos), batch_size):
        batch_file_infos = file_infos[i:i + batch_size]
        try:
            results = Parallel(n_jobs=-1)(delayed(process_image)(file_info) for file_info in batch_file_infos)
            batch_features, batch_labels = zip(*[result for result in results if result[0] is not None])

            features.extend(batch_features)
            labels.extend(batch_labels)
            filenames.extend([file_info[0] for file_info in batch_file_infos if file_info[0] is not None])

            logging.info(f"Processed batch {i // batch_size + 1}/{(len(file_infos) + batch_size - 1) // batch_size}")
        except KeyboardInterrupt:
            logging.warning("Keyboard interrupt detected. Stopping image processing.")
            break

    return np.array(features), np.array(labels), filenames

# Specify paths to dataset folders and CSV file
train_folder = r'C:\Users\Prashant kumar\PycharmProjects\pythonProject7\train'
test_folder = r'C:\Users\Prashant kumar\PycharmProjects\pythonProject7\test1'
csv_file = r'C:\Users\Prashant kumar\PycharmProjects\pythonProject7\sampleSubmission.csv'

# Check if train_folder exists
if not os.path.exists(train_folder):
    logging.error(f"Train folder does not exist: {train_folder}")
    raise SystemExit

# Check if test_folder exists
if not os.path.exists(test_folder):
    logging.error(f"Test folder does not exist: {test_folder}")
    raise SystemExit

# Step 1: Load and Prepare the Data
logging.info("Loading and extracting features from training data...")
X_train, y_train, _ = load_images_and_extract_features(train_folder, limit=999, batch_size=50)
logging.info("Loading and extracting features from test data...")
X_test, y_test, test_filenames = load_images_and_extract_features(test_folder, limit=999, batch_size=50)

# Check if data is loaded correctly
if X_train.size == 0 or X_test.size == 0:
    logging.error("No images were loaded. Please check the folder paths and image files.")
    raise SystemExit

# Step 2: Train the SVM
logging.info("Training the SVM...")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Step 3: Evaluate the SVM
logging.info("Evaluating the SVM...")
y_pred = svm.predict(X_test)

# Step 4: Generate Predictions for Submission
logging.info("Generating submission file...")
submission_df = pd.DataFrame({'id': test_filenames, 'label': y_pred})
submission_df.to_csv(csv_file, index=False, columns=['id', 'label'])

# Step 5: Evaluate Performance (Optional)
logging.info("Evaluating performance...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

logging.info("Process completed.")
