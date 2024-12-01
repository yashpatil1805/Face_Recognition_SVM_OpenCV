import cv2
import numpy as np
import os
import face_recognition
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load and train face recognizer
def train_face_recognizer(image_folder):
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)

    # Encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(known_names)
    
    # Train a classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(known_encodings, labels)
    
    return clf, le

# Load the pre-trained deep learning face detector model
def load_face_detector():
    prototxt = "deploy.prototxt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    return cv2.dnn.readNetFromCaffe(prototxt, model)

# Main function for face recognition
def recognize_faces_in_video():
    clf, le = train_face_recognizer('images')  # Train the recognizer with images in the 'images' folder
    net = load_face_detector()

    # Initialize video capture
    video = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB (required for face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Loop over detected face locations
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Extract the face region
            face = rgb_frame[top:bottom, left:right]
            
            # Get face encodings (Note: face_landmarks might also be useful for certain applications)
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])
            
            if face_encodings:
                # Compare with known faces
                encoding = face_encodings[0]
                matches = clf.predict([encoding])
                name = le.inverse_transform(matches)[0]
                
                # Draw bounding box and name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        
        # Break the loop when 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_in_video()
