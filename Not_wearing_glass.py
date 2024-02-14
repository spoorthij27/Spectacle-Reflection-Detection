import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize the MTCNN detector
detector = MTCNN()

# Placeholder function for glasses detection model prediction
# This function should return True if glasses are detected, and False otherwise
def is_wearing_glasses(eye_region):
    # Implement your model prediction logic here
    # For demonstration, this always returns False (no glasses)
    return False

# Function to detect faces, extract eye regions, and predict glasses wearing
def detect_spectacles(img):
    result = detector.detect_faces(img)
    for face in result:
        bounding_box = face['box']
        keypoints = face['keypoints']

        # Drawing rectangle around the face
        cv2.rectangle(img,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255),
                      2)
        
        # Assuming the eye region can be extracted here for your model
        # For simplicity, this example does not extract an actual eye region
        eye_region = None  # You would extract the actual eye region based on keypoints
        
        # Call your model to check if glasses are present
        wearing_glasses = is_wearing_glasses(eye_region)
        
        if wearing_glasses:
            cv2.putText(img, 'Wearing Glasses', (bounding_box[0], bounding_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            cv2.putText(img, 'No Glasses', (bounding_box[0], bounding_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
    return img

# Main function to capture video from webcam and apply detection
def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, img = cap.read()
        if not ret:
            break

        detected_img = detect_spectacles(img)
        
        cv2.imshow('Spectacles Detection', detected_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()