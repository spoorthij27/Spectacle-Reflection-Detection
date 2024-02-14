import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize the MTCNN detector
detector = MTCNN()

# Function to detect faces and eyes, and to predict glasses wearing
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
        
        # Here you would call your glasses detection model on the eye regions
        # For demonstration, let's just draw circles around the eyes
        cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
        # Add your logic/model to check if glasses are present based on the eye region
        
        # Fake glasses detection (replace with your model's prediction)
        wearing_glasses = True  # Placeholder for actual glasses detection logic
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