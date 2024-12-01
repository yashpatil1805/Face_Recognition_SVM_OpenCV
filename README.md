# Face_Recognition_SVM_OpenCV
This project implements real-time face recognition using OpenCV, face_recognition, and an SVM classifier. It trains a recognizer on labeled images in the images/ folder and detects faces in live video streams, labeling them with names. Customizable and easy to extend for personal datasets.
Files to Include:
face_recognition_svm.py: The main Python script.
deploy.prototxt: Configuration file for the Caffe face detector.
res10_300x300_ssd_iter_140000.caffemodel: Pre-trained weights for the Caffe face detector.
images/: A folder containing labeled images (e.g., John.jpg, Jane.jpg) for training the face recognizer.
.gitignore: To exclude unnecessary files like .pyc, .log, etc.
README.md: Documentation for the project.
Example README.md Content:
Face Recognition with SVM and OpenCV
Description:
This project implements real-time face recognition using OpenCV, face_recognition, and an SVM classifier. It trains a face recognizer on labeled images and detects faces in video streams, associating them with their names.

Features:
Detects and recognizes faces using a pre-trained Caffe face detector and SVM classification.
Labels faces in a live video feed with names based on trained data.
Customizable with your own labeled dataset.
Setup Instructions:
Clone the Repository:


git clone https://github.com/your-username/Face_Recognition_SVM_OpenCV.git
cd Face_Recognition_SVM_OpenCV
Install Dependencies: Ensure Python 3.6+ is installed, then install the required libraries:


pip install opencv-python numpy scikit-learn face_recognition
Prepare the Dataset:

Add labeled images to the images/ folder.
Ensure each image is named after the person in the image (e.g., John.jpg, Jane.png).
Download Required Files:

Add deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel to the repository.
These files can be obtained from the OpenCV GitHub repository.
Usage:
Train the face recognizer and start the live video feed:


python face_recognition_svm.py
The script will:

Detect faces in the video feed.
Recognize and label them based on the trained dataset.
Display the live feed with labeled faces.
Press q to exit the video feed.

File Descriptions:
face_recognition_svm.py: The main script for face recognition.
deploy.prototxt: Defines the architecture of the Caffe face detector.
res10_300x300_ssd_iter_140000.caffemodel: Pre-trained weights for the face detector.
images/: Contains the labeled images for training.
Example Output:
Include a screenshot or example output, such as:

Labeled bounding boxes around faces in the video feed.
Dependencies:
Python 3.6+
OpenCV
NumPy
Scikit-learn
Face_recognition
